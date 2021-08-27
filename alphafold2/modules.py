import torch
from einops import rearrange
from alphafold2.util import Rigid, vec_from_tensor, Rot, BB_fape


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, heads=12, head_dim=16, n_query_points=4, n_value_points=8):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.n_query_points = n_query_points

        node_scalar_attention_inner_dim = heads * head_dim
        node_vector_attention_inner_dim = 3 * n_query_points * heads
        node_vector_attention_value_dim = 3 * n_value_points * heads
        after_final_cat_dim = heads * edge_dim + heads * head_dim + heads * n_value_points * 4

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weight = torch.nn.Parameter(point_weight_init_value)

        self.to_scalar_qkv = torch.nn.Linear(node_dim, 3 * node_scalar_attention_inner_dim, bias=False)
        self.to_vector_qk = torch.nn.Linear(node_dim, 2 * node_vector_attention_inner_dim, bias=False)
        self.to_vector_v = torch.nn.Linear(node_dim, node_vector_attention_value_dim, bias=False)
        self.to_scalar_edge_attention_bias = torch.nn.Linear(edge_dim, heads, bias=False)
        self.final_linear = torch.nn.Linear(after_final_cat_dim, node_dim)

    def forward(self, node_features, edge_features, rigid):
        # Classic attention on nodes
        scalar_qkv = self.to_scalar_qkv(node_features).chunk(3, dim=-1)
        scalar_q, scalar_k, scalar_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), scalar_qkv)
        node_scalar = torch.einsum('b h i d, b h j d -> b h i j', scalar_q, scalar_k) * self.head_dim ** (-1 / 2)

        # Linear bias on edges
        edge_bias = rearrange(self.to_scalar_edge_attention_bias(edge_features), 'b i j h -> b h i j')

        # Reference frame attention
        wc = (2 / self.n_query_points) ** (1 / 2) / 6
        vector_qk = self.to_vector_qk(node_features).chunk(2, dim=-1)
        vector_q, vector_k = map(lambda x: vec_from_tensor(rearrange(x, 'b n (h p d) -> b h n p d', h=self.heads, d=3)),
                                 vector_qk)
        rigid_ = rigid.unsqueeze(1).unsqueeze(-1)  # add head and poitn dimension to rigids

        global_vector_k = rigid_ @ vector_k
        global_vector_q = rigid_ @ vector_q
        global_frame_distance = wc * global_vector_k.unsqueeze(-2).dist(global_vector_k.unsqueeze(-3)).sum(
            -1) * rearrange(self.point_weight, "h -> () h () ()")

        # Combining attentions
        attention_matrix = (3 ** (-1 / 2) * (node_scalar + edge_bias - global_frame_distance)).softmax(-1)

        # Obtaining outputs
        edge_output = (rearrange(attention_matrix, 'b h i j -> b i h () j') * rearrange(edge_features,
                                                                                        'b i j d -> b i () d j')).sum(
            -1)
        scalar_node_output = torch.einsum('b h i j, b h j d -> b i h d', attention_matrix, scalar_v)

        vector_v = vec_from_tensor(
            rearrange(self.to_vector_v(node_features), 'b n (h p d) -> b h n p d', h=self.heads, d=3))
        global_vector_v = rigid_ @ vector_v
        attended_global_vector_v = global_vector_v.map(
            lambda x: torch.einsum('b h i j, b h j p -> b h i p', attention_matrix, x))
        vector_node_output = rigid_.inv() @ attended_global_vector_v
        vector_node_output = torch.stack(
            [vector_node_output.norm(), vector_node_output.x, vector_node_output.y, vector_node_output.z], dim=-1)

        # Concatenate along heads and points
        edge_output = rearrange(edge_output, 'b n h d -> b n (h d)')
        scalar_node_output = rearrange(scalar_node_output, 'b n h d -> b n (h d)')
        vector_node_output = rearrange(vector_node_output, 'b h n p d -> b n (h p d)')

        combined = torch.cat([edge_output, scalar_node_output, vector_node_output], dim=-1)

        return node_features + self.final_linear(combined)


class BackboneUpdate(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()

        self.to_correction = torch.nn.Linear(node_dim, 6)

    def forward(self, node_features):
        # Predict quaternions and translation vector
        rot, t = self.to_correction(node_features).chunk(2, dim=-1)

        # Normalize quaternions
        norm = (1 + rot.pow(2).sum(-1, keepdim=True)).pow(1 / 2)
        b, c, d = (rot / norm).chunk(3, dim=-1)
        a = 1 / norm
        a, b, c, d = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)

        # Make rotation matrix from quaternions
        R = Rot(
            (a ** 2 + b ** 2 - c ** 2 - d ** 2), (2 * b * c - 2 * a * d), (2 * b * d + 2 * a * c),
            (2 * b * c + 2 * a * d), (a ** 2 - b ** 2 + c ** 2 - d ** 2), (2 * c * d - 2 * a * b),
            (2 * b * d - 2 * a * c), (2 * c * d + 2 * a * b), (a ** 2 - b ** 2 - c ** 2 + d ** 2)
        )

        return Rigid(vec_from_tensor(t), R)


class StructureUpdate(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, propagate_rotation_gradient=False, **kwargs):
        super().__init__()
        self.propagate_rotation_gradient = propagate_rotation_gradient

        self.IPA = InvariantPointAttention(node_dim, edge_dim, **kwargs)
        self.norm1 = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(node_dim)
        )
        self.norm2 = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(node_dim)
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(node_dim, 2 * node_dim),  # Pulling these dims out of nowhere
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, node_dim)
        )

        self.backbone_update = BackboneUpdate(node_dim)
        self.FAPE_aux_loss = 0

    def forward(self, node_features, edge_features, rigid_pred, rigid_true=None):
        self.FAPE_aux_loss = 0
        s_i = self.IPA(node_features, edge_features, rigid_pred)
        s_i = self.norm1(s_i)
        s_i = s_i + self.residual(s_i)
        s_i = self.norm2(s_i)
        rigid_new = rigid_pred @ self.backbone_update(s_i)
        if rigid_true is not None:
            self.FAPE_aux_loss += BB_fape(rigid_new, rigid_true)
        if not self.propagate_rotation_gradient:
            rigid_new = Rigid(rigid_new.origin, rigid_new.rot.detach())

        return s_i, rigid_new
