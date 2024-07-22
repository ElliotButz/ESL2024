iric_csv_path = '/home/ebutz/ESL2024/data/full_iric/iric.csv'
iric_path = iric_csv_path
epochs = 15
import sys
sys.path.append('/home/ebutz/ESL2024/code/utils' )

test_ratio = 0.1
val_ratio  = 0.1
# ComplEx embeddings :
hidden_channels = 250
batch_size = 4096
epochs = 30
neg_per_pos = 1 #Number of negatives per positive during training
K = 10 #K from Hit@K

import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.nn import ComplEx
import random

import wandb
from tqdm import tqdm
import constants as c

import torch
import os
import sys
from torch_geometric.utils import to_networkx
sys.path.append('/home/ebutz/ESL2024/code/utils' )
import optuna
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
import torch_geometric.transforms as T
import pickle
import os
import sys
import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import ComplEx
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from nxontology.imports import from_file
import wandb
from play_with_complex import *
from data_utils import *
from train_utils import *
from model_utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from torchmetrics.retrieval import RetrievalMRR, RetrievalHitRate
import play_with_complex as pwc
from work_in_graph import *
from usefull_little_functions import *
import numpy as np

from scipy.stats import ttest_ind
from scipy.stats import t

from nxontology import NXOntology

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def triples_from_csv(path_to_csv, columns_to_use = c.IricNode.features.value):
    """
    Creates triples from a CSV.

    Parameters:
    - path_to_csv (str): Filepath or buffer to the input CSV file.
    - columns_to_use (list): A list of the columns to consider.

    Returns:
    - triples (pandas.DataFrame): Output DataFrame in triple format.
                                  Subjects are index items, predicates are column names from columns_to_use, objects are non-NaN values in columns.
    """

    df = pd.read_csv(filepath_or_buffer=path_to_csv, sep = ',', index_col = 0)
    df.columns = df.columns.str.lower()
    
    # Create a list of triples
    triples = []
    # Drop feature columns
    columns_to_drop = [col for col in columns_to_use if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    df = df.replace({np.nan:None})
    
    for index, row in df.iterrows():
        for column in df.columns:
            if row[column] is not None:
                for predicate in row[column].split('|'):
                    triples.append([index, column, predicate])

    # Create a dataframe from the list of triples
    return pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])

def plot_graph(graph, title="Graph"):
    """
    Plots the given graph using NetworkX.

    :param graph: A NetworkX graph object to be plotted.
    :param title: The title of the graph.
    """
    # Generate positions for the nodes in the graph
    pos = nx.spring_layout(graph, iterations=50, seed=4)
    
    # Draw the graph with node labels and arrows for directed edges
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, arrows=True)
    
    plt.title(title)
    plt.show()

def init_ontology(edges: list, display_output: bool = False, output_name = "ontology"):
    """
    Initializes an ontology using nxontology and adds edges to it.

    Args:
    edges: A list of edges to add to the ontology.
    display_output: If True, the graph is plotted.

    Returns:
    nxo: An NXOntology object with the specified edges.
    """
    # Create an NXOntology object
    nxo: NXOntology[str] = NXOntology()
    nxo.graph.graph["name"] = output_name
    
    # Set the graph attributes for node names
    nxo.set_graph_attributes(node_name_attribute="{node}")
    
    nxo.graph.add_edges_from(edges)

    if display_output:
        plot_graph(nxo.graph, title = output_name)

    return nxo

def DFS(G, v, discovered):
    """
    Performs a Depth-First Search (DFS) on a directed graph.

    Args:
    G (networkx.classes.digraph.DiGraph): The directed graph in which to perform the DFS.
    v: The starting node for the DFS.
    discovered (set): A set to keep track of discovered nodes.

    Returns:
    set: A set of all nodes discovered during the DFS starting from node v.

    Source : https://en.wikipedia.org/wiki/Depth-first_search, july 10 2024.
    """
    discovered.add(v)  # Mark v as discovered

    for w in G.successors(v):    # Traverse all predecessors of v ;
        if w not in discovered:    # If the node w is not yet discovered ;
            DFS(G, w, discovered)  # Recursive DFS call on w ;

    return discovered

def get_leaves_and_ancestors(G):
    """
    Retrieves the leaves and their ancestors from an ontology graph.

    Args:
    G (networkx.classes.digraph.DiGraph): A Directed Graph, should be Acyclic.

    Returns:
    dict: A dictionary where each key is a leaf node and the value is a set of its ancestors.
    """
    # 1 - Listing nodes of the ontology:
    all_nodes: list = list(G.nodes)

    # 2 - For each node, if the node has no children, add node to the list of leaves:
    leaves: list = []

    for node in tqdm(all_nodes, desc="Retrieving leaves from ontology"):

        if len(list(G.predecessors(node))) == 0:
            leaves.append(node)

    # 3 - Associate each leaf and its ancestors in a dict <leaf>: [ancestors]: 
    leaves_and_ancestors = dict()
    for leaf in tqdm(leaves, desc="Retrieving ancestors of each leaf"):
        discovered = set()
        leaves_and_ancestors[leaf] = DFS(G, leaf, discovered)

    return leaves_and_ancestors

def get_distance(graph, node1, node2):
    if node1 == node2:
        return int(0)
    else :
        return len(nx.bidirectional_shortest_path(graph, source = node1, target = node2))
    
def get_similarity(ontology, node1, node2):
    return ontology.similarity(node1,node2).lin

# Extracting triples from original csv :
iric_triples = triples_from_csv(path_to_csv = iric_csv_path)

# Mapping entities and relations to integers ids :
entity_set = set(iric_triples['object']).union(set(iric_triples['subject']))
entity_to_mapping = {entity: int(i) for i, entity in enumerate(entity_set)}
relation_set = set(iric_triples['predicate'])
relation_to_mapping = {relation: int(i) for i, relation in enumerate(relation_set)}

# Triples to mapped triples :
iric_triples['mapped_subject'] = iric_triples['subject'].apply(lambda x: entity_to_mapping[x])
iric_triples['mapped_predicate'] = iric_triples['predicate'].apply(lambda x: relation_to_mapping[x])
iric_triples['mapped_object'] = iric_triples['object'].apply(lambda x: entity_to_mapping[x])
isa_id = relation_to_mapping['is_a']

# Loading Iric as a Data object:
data = load_iric_data('/home/ebutz/ESL2024/data/full_iric/iric.csv', featureless=False)

# Extracting ontology before setting the graph undirected:
GO_edge_index = data[("go", "is_a", "go")]['edge_index']
data  = T.ToUndirected(merge=True)(data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
data = T.RemoveDuplicatedEdges()(data) # Remove duplicated edges

# Defining GO as an nxo.ontology object (for similarity measure):
GO_ontology: nxo.ontology.NXOntology = init_ontology(edges = GO_edge_index.t().tolist())

# Defining GO as an nx.DiGraph object (for neighbourhood extraction. Extracting this graph in a function do not works, idk why.):
GO_nx=GO_ontology.graph

# Mapping :
df = pd.read_csv(iric_path, index_col=0, dtype=str)
df['source_node'] = df.index
df.columns = [x.lower() for x in df.columns]
go_map = get_nodelist(df, 'go')
go_to_idx = {node: i for i, node in enumerate(go_map)}
idx_to_go = {i: node for i, node in enumerate(go_map)}

pwc.map_to_GO        = idx_to_go
pwc.mapped_alt_tails = None
pwc.device           = device

# Preparing the output csv :
GO_leaves_ancestors = get_leaves_and_ancestors(GO_nx)
leaves = list(GO_leaves_ancestors.keys())

distances_and_similarities = pd.DataFrame(columns=['leaf','ancestor', 'distance', 'similarity', 'model_score'])
leaves, ancestors =  [], []
for _, (leaf, leaf_ancestors) in enumerate(GO_leaves_ancestors.items()):
     for leaf_ancestor in leaf_ancestors:
        leaves.append(leaf)
        ancestors.append(leaf_ancestor)
distances_and_similarities['leaf']= leaves
distances_and_similarities['ancestor']= ancestors
distances_and_similarities = distances_and_similarities[distances_and_similarities.distance != 0]
distances_and_similarities = distances_and_similarities.head(10000)
distances_and_similarities['distance']   = distances_and_similarities.apply(lambda row: get_distance(GO_nx, row['leaf'], row['ancestor']), axis=1)
distances_and_similarities['similarity'] = distances_and_similarities.apply(lambda row: get_similarity(GO_ontology, row['leaf'], row['ancestor']), axis=1)

############################# ComplEx training #############################
ontology = iric_triples[iric_triples['predicate']=='is_a']
GO_terms = list(set(list(ontology['mapped_subject']+list(ontology['mapped_object']))))

# Triples to pyg framework :
# Edges index :
heads = list(iric_triples['mapped_subject'])
tails = list(iric_triples['mapped_object'])
edge_index = torch.tensor([heads,tails], dtype=torch.long)
edge_attributes = torch.tensor(iric_triples['mapped_predicate'])
iric_pyg = Data(
                num_nodes = len(entity_set),
                edge_index = edge_index,
                edge_attr = edge_attributes
                )
transform = RandomLinkSplit(
                            num_val = val_ratio,
                            num_test = test_ratio,
                            is_undirected=False,
                            add_negative_train_samples=False,
                            )
train_data, val_data, test_data = transform(iric_pyg)
print("Train, test, val sets look valid :",train_data.validate(raise_on_error=True), test_data.validate(raise_on_error=True), val_data.validate(raise_on_error=True))

# Initiating model :
to_complex = ComplEx(
    num_nodes=train_data.num_nodes,
    num_relations = train_data.edge_index.size()[1],
    hidden_channels=hidden_channels,
).to(device)
to_complex.reset_parameters()
to_complex.to(device)

# Initiaing loader :
head_index = train_data.edge_index[0]
tail_index = train_data.edge_index[1]
rel_type = train_data.edge_attr

loader = to_complex.loader(
    head_index = head_index,
    tail_index = tail_index,
    rel_type = rel_type,
    batch_size=batch_size,
    shuffle=True,
)

# initiating optimizers :
complex_optimizer = optim.Adam(to_complex.parameters())

# Defining test and train functions :
@torch.no_grad()
def test(data, model):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        tail_index=data.edge_index[1],
        rel_type=data.edge_attr,
        batch_size=batch_size, # No need for Tail_Only_ComplEx because one use only 1000 random sample instead of the full dataset.
        k=K, #The k in Hit@k
    )

def train(loader, model, optimizer):
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples

losses = []
for epoch in range(1, epochs+1):
    loss = train(model=to_complex, loader = loader, optimizer=complex_optimizer)
    losses.append(loss)

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    rank, mrr, hit = test(val_data, model=to_complex)
    print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}', f'Val MRR: {mrr:.4f}, Val Hits@10: {hit:.4f}')




#############################

# Making predictions :
heads     = torch.Tensor(distances_and_similarities['leaf'].values).to(torch.long)
relations = torch.Tensor([isa_id]*heads.shape[0]).to(torch.long)
tails     = torch.Tensor(distances_and_similarities['ancestor'].values).to(torch.long)
scores = to_complex(heads, relations, tails)
distances_and_similarities['model_score'] = scores.tolist()

# Saving results :
distances_and_similarities.to_csv(path_or_buf="ontology_undestanding.csv", sep=',')