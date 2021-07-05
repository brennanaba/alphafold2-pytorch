import torch
from einops import rearrange

class InvariantPointAttention(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, heads = 12, head_dim = 16, n_query_points = 4, n_value_points = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.n_query_points = n_query_points

        node_scalar_attention_inner_dim = heads*head_dim
        node_vector_attention_inner_dim = 3*n_query_points*heads
        node_vector_attention_value_dim = 3*n_value_points*heads
        after_final_cat_dim = heads*edge_dim + heads*head_dim + heads*n_value_points*4


        self.to_scalar_qkv = torch.nn.Linear(node_dim, 3*node_scalar_attention_inner_dim, bias = False)
        self.to_vector_qk = torch.nn.Linear(node_dim, 2*node_vector_attention_inner_dim, bias = False)
        self.to_vector_v = torch.nn.Linear(node_dim, node_vector_attention_value_dim, bias = False)
        self.to_scalar_edge_attention_bias = torch.nn.Linear(edge_dim, heads, bias = False)
        self.final_linear = torch.nn.Linear(after_final_cat_dim, node_dim)

    def forward(self,node_features, edge_features, local_reference_frames):

        # Classic attention on nodes
        scalar_qkv = self.to_scalar_qkv(node_features).chunk(3,dim = -1)
        scalar_q, scalar_k, scalar_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), scalar_qkv)
        node_scalar = torch.einsum('b h i d, b h j d -> b h i j', scalar_q, scalar_k)*self.head_dim**(-1/2)

        # Linear bias on edges
        edge_bias = rearrange(self.to_scalar_edge_attention_bias(edge_features), 'b i j h -> b h i j')

        # Reference frame attention
        R,t = local_reference_frames
        wc = (2/9/self.n_query_points)**(1/2)/2
        vector_qk = self.to_vector_qk(node_features).chunk(2,dim = -1)
        vector_q, vector_k = map(lambda t: rearrange(t, 'b n (h p d) -> b h n p d', h = self.heads, d = 3), vector_qk)

        global_vector_k = torch.einsum('b n i j, b h n p j -> b h n p i', R, vector_k) + rearrange(t, 'b n i -> b () n () i')
        global_vector_q = torch.einsum('b n i j, b h n p j -> b h n p i', R, vector_q) + rearrange(t, 'b n i -> b () n () i')
        global_frame_difference =  (rearrange(global_vector_k, 'b h i p d -> b h i () p d') - rearrange(global_vector_q, 'b h j p d -> b h () j p d'))
        global_frame_distance = global_frame_difference.pow(2).sum((-1,-2))*wc

        # Combining attentions
        attention_matrix = (3**(-1/2)*(node_scalar + edge_bias - global_frame_distance)).softmax(-1)

        # Obtaining outputs
        edge_output = (rearrange(attention_matrix, 'b h i j -> b i h () j') * rearrange(edge_features, 'b i j d -> b i () d j')).sum(-1)
        scalar_node_output  = torch.einsum('b h i j, b h j d -> b i h d', attention_matrix, scalar_v)

        vector_v = rearrange(self.to_vector_v(node_features), 'b n (h p d) -> b h n p d', h = self.heads, d = 3)
        global_vector_v = torch.einsum('b n i j, b h n p j -> b h n p i', R, vector_v) + rearrange(t, 'b n i -> b () n () i')
        attended_global_vector_v = torch.einsum('b h i j, b h j p d -> b i h p d', attention_matrix, global_vector_v)
        vector_node_output = torch.einsum('b n i j, b n h p j -> b n h p i',torch.inverse(R),(attended_global_vector_v - rearrange(t, 'b n i -> b n () () i')))

        # Concatenate along heads and points
        edge_output = rearrange(edge_output, 'b n h d -> b n (h d)')
        scalar_node_output = rearrange(scalar_node_output, 'b n h d -> b n (h d)')
        vector_node_length = rearrange(vector_node_output.pow(2).sum(-1).pow(1/2), 'b n h p -> b n (h p)')
        vector_node_output = rearrange(vector_node_output, 'b n h p d -> b n (h p d)')

        combined = torch.cat([edge_output, scalar_node_output, vector_node_output, vector_node_length], dim = -1)

        return node_features + self.final_linear(combined)


class BackboneUpdate(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()

        self.to_correction = torch.nn.Linear(node_dim, 6)

    def forward(self, node_features):
        # Predict quaternions and translation vector
        rot, t = self.to_correction(node_features).chunk(2,dim = -1)

        # Normalize quaternions
        norm = (1 + rot.pow(2).sum(-1, keepdim = True)).pow(1/2)
        b, c, d = (rot/norm).chunk(3, dim = -1)
        a = 1/norm
        a, b, c, d  = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)

        # Make rotation matrix from quaternions
        R = torch.zeros((*node_features.shape[:-1], 3, 3))
        R[..., 0,0], R[..., 0,1], R[..., 0,2] = (a**2 + b**2 - c**2 - d**2), (2*b*c - 2*a*d), (2*b*d + 2*a*c)
        R[..., 1,0], R[..., 1,1], R[..., 1,2] = (2*b*c + 2*a*d), (a**2 - b**2 + c**2 - d**2), (2*c*d - 2*a*b)
        R[..., 2,0], R[..., 2,1], R[..., 2,2] = (2*b*d - 2*a*c), (2*c*d + 2*a*b), (a**2 - b**2 - c**2 + d**2)

        return R,t