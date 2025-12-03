import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


class DummyEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # x: (N, D, D)
        x = x.mean(dim=1)
        return self.linear(x)

# --------- Base MLP Encoder ---------
class MLPEncoder(nn.Module):
    """
    A simple fully-connected encoder for vector features (e.g., morphological data).
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=32):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# --------- CNN Encoder (for image-like inputs, optional) ---------
class CNNEncoder(nn.Module):
    """
    Convolutional encoder for spatial or image-like inputs.
    Useful for 2D/3D maps of connectivity or region maps.
    """
    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x):
        # input x: (B, 1, H, W)
        x = self.conv(x)  # (B, 16, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 16)
        return self.fc(x)  # (B, out_dim)



# --------- Transformer Encoder (optional for sequence modeling) ---------
class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequential input like region-wise features (e.g., ordered ROI vectors).
    """
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, output_dim=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (B, S, D) → sequence of features
        x = self.input_proj(x)  # (B, S, embed_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (S, B, E)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Mean pooling over sequence
        return self.fc_out(x)  # (B, output_dim)


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder for node-level attention aggregation.

    Parameters:
    - in_feats: Input feature dimension per node
    - hidden_dim: Hidden dimension for GAT heads
    - out_dim: Final output embedding dimension per subject
    - num_heads: Number of attention heads per GAT layer
    - num_layers: Number of GAT layers
    - dropout: Dropout rate between layers
    """

    def __init__(self, in_feats, hidden_dim=64, out_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, hidden_dim, num_heads, activation=F.relu))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads, activation=F.relu))
        self.layers.append(dglnn.GATConv(hidden_dim * num_heads, out_dim, num_heads=1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        for layer in self.layers[:-1]:
            h = layer(g, h).flatten(1)
            h = self.dropout(h)
        h = self.layers[-1](g, h).squeeze(1)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder with mean neighbor aggregation.

    Parameters:
    - in_feats: Input feature dimension per node
    - hidden_dim: Intermediate hidden dimension
    - out_dim: Output subject-level embedding
    - num_layers: Number of SAGEConv layers
    - dropout: Dropout probability
    """
    def __init__(self, in_feats, hidden_dim=64, out_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, hidden_dim, aggregator_type='mean'))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))
        self.layers.append(dglnn.SAGEConv(hidden_dim, out_dim, aggregator_type='mean'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        for layer in self.layers[:-1]:
            h = F.relu(layer(g, h))
            h = self.dropout(h)
        h = self.layers[-1](g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


class GINEncoder(nn.Module):
    """
    GIN encoder capturing isomorphic neighborhood structures.

    Parameters:
    - in_feats: Input feature size per node
    - hidden_dim: MLP hidden units
    - out_dim: Output dimension
    - num_layers: Number of GIN layers
    - dropout: Dropout rate
    """
    def __init__(self, in_feats, hidden_dim=64, out_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(nn.Linear(in_feats, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            elif i == num_layers - 1:
                mlp = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
            else:
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

            self.layers.append(dglnn.GINConv(mlp, learn_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else out_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(g, h)
            h = self.batch_norms[i](h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

