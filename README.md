# alphafold2-pytorch

## Usage

I have written this mostly for my own use and experimentation. If you want to use it feel free

#### To use Invariant Point Attention model:

```python
node_features = torch.randn((1,20,23))
edge_features = torch.randn((1,20,20,5))
atom_coordinates = torch.randn((1,20,3,3)) # 3 atoms per residue
rigid_residues = rigid_from_tensor(atom_coordinates)

model = InvariantPointAttention(node_dim = 23, edge_dim = 5)
updated_node_features = model(node_features, edge_features, rigid_residues) # (1,20,23)
```

#### To use Structure Update model:


```python
node_features = torch.randn((1,20,23))
edge_features = torch.randn((1,20,20,5))
atom_coordinates = torch.randn((1,20,3,3)) # 3 atoms per residue
rigid_residues = rigid_from_tensor(atom_coordinates)

model = StructureUpdate(node_dim = 23, edge_dim = 5)
updated_node_features, updated_rigid_residues = model(node_features)
```
