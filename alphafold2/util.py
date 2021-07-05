import torch
import numpy as np
import plotly.graph_objects as go
from einops import rearrange

def BB_atom_coords_to_local_reference_frame(coors):
    """ Computes residue reference frame from N-CA-C atoms.
    Assumes coords is of shape (n_residue, atom_type, coordinate) and atoms are ordered as N-CA-C.
    """
    R = torch.zeros_like(coors)
    t = coors[:,1]
    v1 = coors[:,2] - t
    v2 = coors[:,0] - t
    R[:,0] = v1/v1.norm(p=2, dim=-1, keepdim=True)
    u2 = v2 - R[:,0] * (R[:,0] * v2).sum(-1, keepdim = True)
    R[:,1] = u2/u2.norm(p=2, dim=-1, keepdim=True)
    R[:,2] = torch.cross(R[:,0], R[:,1])

    return R, t

def rotation_matrix_z_axis(alpha):
    cos = torch.cos(alpha)
    sin = torch.sin(alpha)
    return torch.tensor([
                         [cos , sin, 0.0],
                         [-sin, cos, 0.0],
                         [0.0 , 0.0, 1.0]
    ], dtype = alpha.dtype, device = alpha.device)

def rotation_around_vector(x,alpha = 1.92):
    A = torch.zeros((*x.shape, 3),dtype = x.dtype, device = x.device)
    A[..., 0,1], A[..., 0,2], A[..., 1,0] = -x[...,2], x[...,1], x[...,2]
    A[..., 1,2], A[..., 2,0], A[..., 2,1] = -x[...,0], -x[...,1], x[...,0]

    return alpha.sin()*A + (2*(alpha/2).sin().pow(2)* (A@A))

def local_reference_frame_to_atom_coords(R,t):
    """ Transforms local reference frames into atom coords
    CA-C distance is fixed to 1.52
    CA-N distance is fixed to 1.46
    N_CA_C angle is fixed to 110 degrees (~1.92 radians)
    """
    coords = torch.zeros_like(R)
    coords[:,1] = t
    coords[:,2] = 1.52*(R[:,0]) + t
    N_CA_C_angle = torch.tensor(1.92, device = R.device, dtype = R.dtype)
    R_z = rotation_matrix_z_axis(N_CA_C_angle)
    coords[:,0] = 1.46*(R_z @ R)[:,0] + t
    return coords

def plot_reference_frames(R,t):
    data = []
    R0,t0 = 3*R.cpu().detach().numpy(), t.cpu().detach().numpy()
    for i,r in enumerate(R0):
        xs = np.zeros((3,4))
        xs[:] += t0[i, :, None]
        xs[:,1:] += r
        mesh = go.Mesh3d(
            x=xs[0],
            y=xs[1],
            z=xs[2],
            # Intensity of each vertex, which will be interpolated and color-coded
            # i, j and k give the vertices of triangles
            # here we represent the 4 triangles of the tetrahedron surface
            i=[0,0,1],
            j=[1,2,2],
            k=[3,3,3],
            opacity=0.4,
            color='blue'
            )
        data.append(mesh)
    return data

def compute_FAPE_loss(pred_reference_frames, true_reference_frames, pred_positions, true_positions):
    x_pred, x_true = pred_positions, true_positions
    R_pred, t_pred = pred_reference_frames
    R_true, t_true = true_reference_frames

    x_local_pred = torch.einsum('b n i j, b p j -> b n p i', R_pred, x_pred) + rearrange(t_pred, 'b n i -> b n () i')
    x_local_true = torch.einsum('b n i j, b p j -> b n p i', R_true, x_true) + rearrange(t_true, 'b n i -> b n () i')

    diff = ((x_local_pred - x_local_true).pow(2).sum(-1) + 1e-4).pow(1/2)

    loss = diff.clamp(max = 10).mean()/10

    return loss