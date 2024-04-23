import sys

sys.path.append( '/home/ebutz/ESL2024/code/utils' )

from train_utils import *

from torch_geometric.nn import *
from torch_geometric.nn.conv import *
from torch_geometric.nn.norm import *
import torch


from copy import deepcopy

def get_gnn_layers(config):
    """
    Returns possible GNN layer architectures.
    
    Parameters
    ----------
    config : object
        An object containing the configuration parameters for the model. This should include the number of layers, the hidden dimension, 
        the number of attention heads (if required), the dropout rate, and the normalization layer.
    
    Returns
    -------
    layers : dict
        A dict containing the layer architectures.
    """
    
    if not hasattr(config, 'attention_heads'):
        config.attention_heads = 1

    gnn_layers = {
        "GATv2Conv" : GATv2Conv(in_channels= (-1,-1), out_channels=config.hidden_channels, heads=config.attention_heads, dropout=config.dropout, concat=False, add_self_loops=False),
        "TransformerConv" : TransformerConv(in_channels= (-1,-1), out_channels=config.hidden_channels, heads=config.attention_heads, concat=False, dropout=config.dropout),
        "MFConv" : MFConv(in_channels= (-1,-1), out_channels=config.hidden_channels, max_degree=10),
        "SAGEConv" : SAGEConv(in_channels= (-1,-1), out_channels=config.hidden_channels),
        "GraphConv" : GraphConv(in_channels= (-1,-1), out_channels=config.hidden_channels),
        "ResGatedGraphConv" : ResGatedGraphConv(in_channels= (-1,-1), out_channels=config.hidden_channels),
        "LEConv" : LEConv(in_channels= (-1,-1), out_channels=config.hidden_channels, aggr=config.aggregation),
        "GeneralConv" : GENConv(in_channels= (-1,-1), out_channels=config.hidden_channels, aggr=config.aggregation, learn_t=True, learn_p=True)
    }

    return gnn_layers

def get_homogeneous_gnn_layers(config):
    """
    Returns possible GNN layer architectures.
    
    Parameters
    ----------
    config : object
        An object containing the configuration parameters for the model. This should include the number of layers, the hidden dimension, 
        the number of attention heads (if required), the dropout rate, and the normalization layer.
    
    Returns
    -------
    layers : dict
        A dict containing the layer architectures.
    """
    
    if not hasattr(config, 'attention_heads'):
        config.attention_heads = 1

    gnn_layers = {
        "GATv2Conv" : GATv2Conv(in_channels=-1, out_channels=config.hidden_channels, heads=config.attention_heads, dropout=config.dropout, concat=False, add_self_loops=False),
        "TransformerConv" : TransformerConv(in_channels=-1, out_channels=config.hidden_channels, heads=config.attention_heads, concat=False, dropout=config.dropout),
        "MFConv" : MFConv(in_channels=-1, out_channels=config.hidden_channels, max_degree=10),
        "SAGEConv" : SAGEConv(in_channels=-1, out_channels=config.hidden_channels),
        "GraphConv" : GraphConv(in_channels=-1, out_channels=config.hidden_channels),
        "ResGatedGraphConv" : ResGatedGraphConv(in_channels=-1, out_channels=config.hidden_channels),
        "LEConv" : LEConv(in_channels=-1, out_channels=config.hidden_channels, aggr=config.aggregation),
        "GeneralConv" : GENConv(in_channels=-1, out_channels=config.hidden_channels, aggr=config.aggregation, learn_t=True, learn_p=True)
    }

    return gnn_layers

def get_norm_layers(config, nb_nodetypes):
    """
    Returns possible normalization layers.

    Parameters
    ----------
    config : object
        An object containing the configuration parameters for the model. This should include the hidden dimension, 
        alongside the number of node types in the graph.
    
    Returns
    -------
    norm_layers : dict
        A dict containing the normalization layers.
    """


    # Normalization layers pool
    norm_layers = {
        "BatchNorm" : BatchNorm(config.hidden_channels),
        "LayerNorm" : LayerNorm(config.hidden_channels),
        "InstanceNorm" : InstanceNorm(config.hidden_channels),
        "GraphNorm" : GraphNorm(config.hidden_channels),
        "GraphSizeNorm" : GraphSizeNorm(),
        "MeanSubtractionNorm" : MeanSubtractionNorm(),
        "DiffGroupNorm" : DiffGroupNorm(config.hidden_channels, groups=nb_nodetypes),
        "None" : None
    }

    return norm_layers

# Decoder (Link Prediction module)
class Linear(torch.nn.Module):
    """
    A class that applies a linear transformation to the node pair embeddings obtained from a GNN encoder for link prediction.

    Parameters
    ----------
    hidden_channels : int
        The size of the hidden layer (i.e the embedding size of nodes).
    dropout : float
        The dropout rate for the dropout layer.
    norm : callable
        A normalization function to apply to the node embeddings after the first linear transformation. If None, no normalization is applied.

    Attributes
    ----------
    lin1 : torch.nn.Linear
        The first linear transformation layer.
    lin2 : torch.nn.Linear
        The second linear transformation layer.
    dropout : torch.nn.Dropout
        The dropout layer.
    norm : callable
        The normalization function to apply to the node embeddings.

    Methods
    -------
    forward(z_dict, sampled_data):
        Apply the linear transformations to the node embeddings.

    """

    def __init__(self, config, norm=None):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * config.hidden_channels, config.hidden_channels)
        self.lin2 = torch.nn.Linear(config.hidden_channels, 1)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.norm = norm
    def forward(self, z_dict, sampled_data, config):
        src_idx, dst_idx = get_sampled_edge_indices(sampled_data, config)

        x = torch.cat([z_dict[config.labels['tail']][src_idx], z_dict[config.labels['tail']][dst_idx]], dim=-1) # Concatenate node embeddings
        x = self.lin1(x)
        x = torch.relu(x)
        if self.norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x.view(-1)
    
class HomoLinear(torch.nn.Module):
    """
    A class that applies a linear transformation to the node pair embeddings obtained from a GNN encoder for link prediction.

    Parameters
    ----------
    hidden_channels : int
        The size of the hidden layer (i.e the embedding size of nodes).
    dropout : float
        The dropout rate for the dropout layer.
    norm : callable
        A normalization function to apply to the node embeddings after the first linear transformation. If None, no normalization is applied.

    Attributes
    ----------
    lin1 : torch.nn.Linear
        The first linear transformation layer.
    lin2 : torch.nn.Linear
        The second linear transformation layer.
    dropout : torch.nn.Dropout
        The dropout layer.
    norm : callable
        The normalization function to apply to the node embeddings.

    Methods
    -------
    forward(z_dict, sampled_data):
        Apply the linear transformations to the node embeddings.

    """

    def __init__(self, config, norm=None):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * config.hidden_channels, config.hidden_channels)
        self.lin2 = torch.nn.Linear(config.hidden_channels, config.hidden_channels, 1)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.norm = norm
    def forward(self, x, sampled_data, config):
        src_idx, dst_idx = get_sampled_edge_indices(sampled_data, config)
        
        x = torch.cat([x[src_idx], x[dst_idx]], dim=-1) # Concatenate node embeddings
        x = self.lin1(x)
        x = torch.relu(x)
        if self.norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x.view(-1)

class DotProduct(torch.nn.Module):
    def forward(self, z_dict, sampled_data, config):
        row, col = get_sampled_edge_indices(sampled_data, config)
        dot_product = torch.einsum('ij,ij->i', z_dict[config.labels['tail']][row], z_dict[config.labels['tail']][col]) # Compute dot product
        return dot_product
    
class CosineSimilarity(torch.nn.Module):
    def forward(self, z_dict, sampled_data, config):
        row, col = get_sampled_edge_indices(sampled_data, config)
        cos_sim = F.cosine_similarity(z_dict[config.labels['tail']][row], z_dict[config.labels['tail']][col], dim=1) # Compute cosine similarity
        return cos_sim

class Model(torch.nn.Module):
    """
    A Model class that combines an encoder and a decoder for processing graph data.

    Parameters
    ----------
    config : object
        A configuration object that contains the parameters for the model. It should have the following attributes:
        - num_layers: The number of layers in the encoder.
        - gnn_layer: The index of the GNN layer to use in the encoder.
        - norm: The index of the normalization function to use in the encoder and decoder.
        - aggregation: The aggregation method to use in the encoder. It should be one of "sum", "mean", "min", "max", or "mul".
        - hidden_channels: The size of the hidden layer in the decoder.
        - dropout: The dropout rate for the decoder.
    norm_layers : dict of callable
        A list of normalization functions. The function to use is selected by the 'norm' attribute of the config object.
    gnn_layers : dict of torch.nn.Module
        A list of GNN layers. The layer to use is selected by the 'gnn_layer' attribute of the config object.

    Attributes
    ----------
    encoder : HeteroGNN
        The encoder part of the model.
    decoder : Linear
        The decoder part of the model.

    Methods
    -------
    forward(sampled_data):
        Apply the encoder and decoder to the sampled data.

    """

    def __init__(self, config, data, norm_layers, gnn_layers):
        super().__init__()
        self.encoder = HeteroGNN(num_layers=config.num_layers, gnn_layer=gnn_layers[config.gnn_layer], norm=norm_layers[config.norm])
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr=config.aggregation)
        self.decoder = Linear(config, norm=norm_layers[config.norm])

    def forward(self, sampled_data, config):
        z_dict = self.encoder(sampled_data.x_dict, sampled_data.edge_index_dict)
        return self.decoder(z_dict, sampled_data, config)


class HeteroGNN(torch.nn.Module):
    """
    A Heterogeneous Graph Neural Network (HeteroGNN) class that applies multiple layers of a specified GNN layer to node features.

    Parameters
    ----------
    num_layers : int
        The number of GNN layers to apply.
    gnn_layer : torch.nn.Module
        The GNN layer to apply. This layer should be capable of handling bipartite graphs with differently sized node features.
    norm : callable
        A normalization function to apply to the node features after each GNN layer. If None, no normalization is applied.

    Attributes
    ----------
    convs : torch.nn.ModuleList
        A list of GNN layers to sequentially apply.
    norm : callable
        The normalization function to apply to the node features.

    Methods
    -------
    forward(x_dict, edge_index_dict):
        Apply the GNN layers to the node features.

    """
    def __init__(self, num_layers, gnn_layer, norm=None):
        super().__init__()

        # When working on bipartite graph with differently sized node features, layers need both lazy init and bipartite handling properties (see PyG GNN cheatsheet)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(deepcopy(gnn_layer))
        self.norm = norm

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            if self.norm:
                x_dict = self.norm(x_dict)
            x_dict= x_dict.relu()
        return x_dict
    
class HomegeneousModel(torch.nn.Module):
    """
    A Model class that combines an encoder and a decoder for processing graph data.

    Parameters
    ----------
    config : object
        A configuration object that contains the parameters for the model. It should have the following attributes:
        - num_layers: The number of layers in the encoder.
        - gnn_layer: The index of the GNN layer to use in the encoder.
        - norm: The index of the normalization function to use in the encoder and decoder.
        - aggregation: The aggregation method to use in the encoder. It should be one of "sum", "mean", "min", "max", or "mul".
        - hidden_channels: The size of the hidden layer in the decoder.
        - dropout: The dropout rate for the decoder.
    norm_layers : dict of callable
        A list of normalization functions. The function to use is selected by the 'norm' attribute of the config object.
    gnn_layers : dict of torch.nn.Module
        A list of GNN layers. The layer to use is selected by the 'gnn_layer' attribute of the config object.

    Attributes
    ----------
    encoder : HeteroGNN
        The encoder part of the model.
    decoder : Linear
        The decoder part of the model.

    Methods
    -------
    forward(sampled_data):
        Apply the encoder and decoder to the sampled data.

    """

    def __init__(self, config, data, norm_layers, gnn_layers):
        super().__init__()
        self.encoder = HomoGNN(num_layers=config.num_layers, homo_gnn_layer=gnn_layers[config.gnn_layer], norm=norm_layers[config.norm])
        self.decoder = HomoLinear(config, norm=norm_layers[config.norm])

    def forward(self, sampled_data, config):
        x = self.encoder(sampled_data.x, sampled_data.edge_index)
        return self.decoder(x, sampled_data, config)

class HomoGNN(torch.nn.Module):
    """
    A Homogeneous Graph Neural Network class that applies multiple layers of a specified GNN layer to node features.

    Parameters
    ----------
    num_layers : int
        The number of GNN layers to apply.
    gnn_layer : torch.nn.Module
        The GNN layer to apply.
    norm : callable
        A normalization function to apply to the node features after each GNN layer. If None, no normalization is applied.

    Attributes
    ----------
    convs : torch.nn.ModuleList
        A list of GNN layers to sequentially apply.
    norm : callable
        The normalization function to apply to the node features.

    Methods
    -------
    forward(x_dict, edge_index_dict):
        Apply the GNN layers to the node features.

    """
    def __init__(self, num_layers, homo_gnn_layer, norm=None):
        super().__init__()

        # When working on bipartite graph with differently sized node features, layers need both lazy init and bipartite handling properties (see PyG GNN cheatsheet)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(deepcopy(homo_gnn_layer))
        self.norm = norm

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.norm:
                x = self.norm(x)
            x = x.relu()
        return x