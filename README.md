# alphafold2-pytorch

## Usage

#### To use Invariant Point Attention model:

```python
node_features = torch.randn((1,20,23))
edge_features = torch.randn((1,20,20,5))
R = torch.randn((1,20,3,3))
t = torch.randn((1,20,3))


model = InvariantPointAttention(node_dim = 23, edge_dim = 5)
updated_node_features = model(node_features, edge_features, (R,t)) # (1,20,23)
```

There is an unexplained lambda in the code that I have set to 1.

#### To use Backbone Update model:


```python
node_features = torch.randn((1,20,23))

model = BackboneUpdate(node_dim = 23)
R,t = model(node_features)
```
