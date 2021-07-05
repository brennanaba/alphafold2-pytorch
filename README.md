# alphafold2-pytorch

## Usage

To use Invariant Point Attention:

```python
node_features = torch.randn((1,20,23))
edge_features = torch.randn((1,20,20,5))
R = torch.randn((1,20,3,3))
t = torch.randn((1,20,3))


model = InvariantPointAttention(23, 5)
updated_node_features = model(node_features, edge_features, (R,t)) # (1,20,23)
```
