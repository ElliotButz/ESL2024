import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch


def nx_successors(graph, node):
    """
    Finds successors of a node in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - node: The node to find successors for.

    Returns:
    - successors (set): Set of successor nodes.
    """
    return set([n for n in graph.successors(node)])

def nx_predecessors(graph, node):
    """
    Finds predecessors of a node in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - node: The node to find predecessors for.

    Returns:
    - predecessors (set): Set of predecessor nodes.
    """
    return set([n for n in graph.predecessors(node)])

def nx_neighbors(graph, node):
    """
    Finds neighbors of a node in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - node: The node to find neighbors for.

    Returns:
    - neighbors (set): Set of neighbor nodes.
    """
    return nx_predecessors(graph, node).union(nx_successors(graph, node))

def successors_for_nodes(graph, nodes):
    """
    Finds successors for a list of nodes in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - nodes (list): List of nodes to find successors for.

    Returns:
    - successors (set): Set of successor nodes for all nodes in the list.
    """
    successors = set()
    for node in nodes:
        successors.update(nx_successors(graph, node))
    return successors

def predecessors_for_nodes(graph, nodes):
    """
    Finds predecessors for a list of nodes in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - nodes (list): List of nodes to find predecessors for.

    Returns:
    - predecessors (set): Set of predecessor nodes for all nodes in the list.
    """
    predecessors = set()
    for node in nodes:
        predecessors.update(nx_predecessors(graph, node))
    return predecessors

def neighbors_for_nodes(graph, nodes):
    """
    Finds neighbors for a list of nodes in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - nodes (list): List of nodes to find neighbors for.

    Returns:
    - neighbors (set): Set of neighbor nodes for all nodes in the list.
    """
    neighbors = set()
    for node in nodes:
        neighbors.update(nx_neighbors(graph, node))
    return neighbors

def nx_successors(graph, node):
    return set([n for n in graph.successors(node)])

def nx_predecessors(graph, node):
    return set([n for n in graph.predecessors(node)])

def nx_neighbors(graph, node):
    return nx_predecessors(graph, node).union(nx_successors(graph, node))

def successors_for_nodes(graph, nodes):
    successors = set()
    for node in nodes:
        successors.update(nx_successors(graph, node))
    return successors

def predecessors_for_nodes(graph, nodes):
    predecessors = set()
    for node in nodes:
        predecessors.update(nx_predecessors(graph, node))
    return predecessors

def neighbors_for_nodes(graph, nodes):
    neighbors = set()
    for node in nodes:
        neighbors.update(nx_neighbors(graph, node))
    return neighbors

def get_family(ontology_edge_index, max_dist):

    """
    Constructs a family graph up to a certain distance from a given ontology edge index.

    Parameters:
    - ontology_edge_index (torch.Tensor): Edge index tensor representing the ontology graph.
                                          /!\ Graph sould probably be oriented !
    - max_dist (int): Maximum distance to traverse to construct the family graph.
                    

    Returns:
    - df (pandas.DataFrame): DataFrame containing information about the family graph.
                            Columns:
                                - 0-neighbors: The departure node of the row.
                                - 0-children: Set of immediate children of the departure node.
                                - 0-parents: Set of immediate parents of the departure node.
                                - {dist}-neighbors: Set of neighbors at distance 'dist' from the departure node.
                                - {dist}-parents: Set of parents at distance 'dist' from the departure node.
                                - {dist}-children: Set of children at distance 'dist' from the departure node.
                                - lineage: Set of all ancestors and descendants up to distance 'max_dist' for the departure node.
                                - family_without_lineage: Set of family members excluding lineage for the departure node.
    """

    # Init Graph
    G = nx.DiGraph()
    for s, d in zip(ontology_edge_index[0].tolist(), ontology_edge_index[1].tolist()):
        G.add_edge(s, d)

    # Listing terms
    all_terms =  torch.cat([ontology_edge_index[0], ontology_edge_index[1]]).unique().tolist()

    df = pd.DataFrame(all_terms, columns=['0-neighbors'])
    df['departure']   = df.apply(lambda row : set([int(row['0-neighbors'])]), axis = 1)
    df['0-neighbors'] = df.apply(lambda row :  neighbors_for_nodes(graph = G, nodes = row['departure']), axis=1)
    df['0-children']  = df.apply(lambda row : successors_for_nodes(graph = G, nodes = row['departure']), axis=1)
    df['0-parents']   = df.apply(lambda row:predecessors_for_nodes(graph = G, nodes = row['departure']), axis=1)

    for dist in range(1, max_dist+1):
        step = f'Distance {dist}/{max_dist}'
        print(step)

        tqdm.pandas(desc=f'Making dist {dist}-neighbors')
        if dist>1 :
            df[f'{dist}-neighbors'] = df.progress_apply(lambda row : neighbors_for_nodes(graph = G, nodes = row[f'{dist-1}-neighbors'])-row[f'{dist-1}-neighbors'],
                                                    axis = 1)
        else :
            df[f'{dist}-neighbors'] = df.progress_apply(lambda row : neighbors_for_nodes(graph = G, nodes = row[f'{dist-1}-neighbors'])-row[f'departure'],
                                                    axis = 1)
        
        tqdm.pandas(desc=f'Making dist {dist}-parents  ')
        df[f'{dist}-parents'] = df.progress_apply(lambda row : predecessors_for_nodes(graph = G, nodes = row[f'{dist-1}-parents']),
                                                    axis = 1)

        tqdm.pandas(desc=f'Making dist {dist}-children ')
        df[f'{dist}-children'] = df.progress_apply(lambda row : successors_for_nodes(graph = G, nodes = row[f'{dist-1}-children']),
                                                    axis = 1)
        
        # display(df)
    
    step = f'{dist}-familiy acquired.'
    print(step)

    tqdm.pandas(desc=f'Calculating lineage          ')
    df['lineage'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-children'] for dist in range(0, max_dist+1)]+
                                                             [row[f'{dist}-parents'] for dist in range(0, max_dist+1)] +
                                                             [row['departure']])
                                      ,axis=1)
    
    tqdm.pandas(desc=f'Calculating familiy - lineage')
    df['family_without_lineage'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-neighbors'] for dist in range(1, max_dist+1)]) - row['lineage']
                                                     ,axis=1)
    
    # Re-ordering columns :
    dists = range(0, max_dist)
    df = df[['departure'] + [f'{dist}-children' for dist in dists] + [f'{dist}-parents' for dist in dists] +[f'{dist}-neighbors' for dist in dists] + ['lineage', 'family_without_lineage']]

    return df

def make_dict_from_columns(df, col_keys, col_values, tensorize_values = False):
    """
    Creates a dictionary from specified columns of a DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame from which to create the dictionary.
    - col_keys (str or list): Column name(s) to use as keys in the dictionary.
    - col_values (str or list): Column name(s) to use as values in the dictionary.

    Returns:
    - result_dict (dict): Dictionary created from specified columns.
    """
    if not tensorize_values :
        return df.set_index(col_keys)[col_values].to_dict()
    else :
        df = df.set_index(col_keys)[col_values]
        df[col_values].apply(lambda row : row[col_values])