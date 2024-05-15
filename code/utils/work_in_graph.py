import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch
# import usefull_little_functions



def nx_predecessors(graph, node):
    """
    Finds predecessors of a node in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - node: The node to find predecessors for.

    Returns:
    - predecessors (set): Set of successor nodes.
    """
    return set([n for n in graph.predecessors(node)])

def nx_successors(graph, node):
    """
    Finds successors of a node in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - node: The node to find successors for.

    Returns:
    - successors (set): Set of predecessor nodes.
    """
    return set([n for n in graph.successors(node)])

def nx_neighbors(graph, node):
    """
    Finds neighbors of a node in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - node: The node to find neighbors for.

    Returns:
    - neighbors (set): Set of neighbor nodes.
    """
    return nx_successors(graph, node).union(nx_predecessors(graph, node))

def predecessors_for_nodes(graph, nodes):
    """
    Finds predecessors for a list of nodes in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - nodes (list): List of nodes to find predecessors for.

    Returns:
    - predecessors (set): Set of successor nodes for all nodes in the list.
    """
    predecessors = set()
    for node in nodes:
        predecessors.update(nx_predecessors(graph, node))
    return predecessors

def successors_for_nodes(graph, nodes):
    """
    Finds successors for a list of nodes in a graph.

    Parameters:
    - graph (networkx.Graph): The graph object.
    - nodes (list): List of nodes to find successors for.

    Returns:
    - successors (set): Set of predecessor nodes for all nodes in the list.
    """
    successors = set()
    for node in nodes:
        successors.update(nx_successors(graph, node))
    return successors

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

def nx_predecessors(graph, node):
    return set([n for n in graph.predecessors(node)])

def nx_successors(graph, node):
    return set([n for n in graph.successors(node)])

def nx_neighbors(graph, node):
    return nx_successors(graph, node).union(nx_predecessors(graph, node))

def predecessors_for_nodes(graph, nodes):
    predecessors = set()
    for node in nodes:
        predecessors.update(nx_predecessors(graph, node))
    return predecessors

def successors_for_nodes(graph, nodes):
    successors = set()
    for node in nodes:
        successors.update(nx_successors(graph, node))
    return successors

def neighbors_for_nodes(graph, nodes):
    neighbors = set()
    for node in nodes:
        neighbors.update(nx_neighbors(graph, node))
    return neighbors

def get_family(ontology_edge_index, max_dist):
    """
    Constructs a family graph up to a certain distance from a given ontology edge index.

    Parameters:
    - ontology_edge_index (torch.Tensor) : Edge index tensor representing the ontology graph.
    - max_dist (int)                     : Maximum distance to traverse to construct the family graph.      

    Returns:
    - df (pandas.DataFrame): DataFrame containing information about the family graph.
                                Columns:
                                    - source           : The source node of the row.
                                    - {dist}-neighbors : Set of neighbors at distance 'dist' from the source node.
                                    - {dist}-parents   : Set of parents at distance 'dist' from the source node.
                                    - {dist}-children  : Set of children at distance 'dist' from the source node.
                                    - family           : Set of all neighbors up to distance 'max_dist' for the source node.
                                    - family_without_children : Set of family members excluding children for the source node.
    """

    # Init Graph
    G = nx.DiGraph()
    for s, d in zip(ontology_edge_index[0].tolist(), ontology_edge_index[1].tolist()):
        G.add_edge(s, d)

    if not nx.is_directed(G):
        raise Exception("Graph should be directed.")

    # Listing terms
    all_terms =  torch.cat([ontology_edge_index[0], ontology_edge_index[1]]).unique().tolist()
    # print(all_terms)
    df = pd.DataFrame(all_terms, columns=['source'])
    df['source']      = df.apply(lambda row : set([int(row['source'])]), axis = 1)
    df['1-neighbors'] = df.apply(lambda row :    neighbors_for_nodes(graph = G, nodes = row['source']), axis=1)
    df['1-children']  = df.apply(lambda row : predecessors_for_nodes(graph = G, nodes = row['source']), axis=1)
    df['1-parents']   = df.apply(lambda row :   successors_for_nodes(graph = G, nodes = row['source']), axis=1)

    for dist in range(2, max_dist+1):

        step = f'Distance {dist}/{max_dist}'
        print(step)

        tqdm.pandas(desc=f'Making dist {dist}-neighbors ')
        df[f'{dist}-neighbors'] = df.progress_apply(lambda row : neighbors_for_nodes(graph = G, nodes = row[f'{dist-1}-neighbors'])-set.union(*[row[f'{d}-neighbors'] for d in range(1,dist)])-row['source'],axis = 1)

        tqdm.pandas(desc=f'Making dist {dist}-parents   ')
        df[f'{dist}-parents'] = df.progress_apply(lambda row : successors_for_nodes(graph = G, nodes = row[f'{dist-1}-parents']),axis = 1)

        tqdm.pandas(desc=f'Making dist {dist}-children  ')
        df[f'{dist}-children'] = df.progress_apply(lambda row : predecessors_for_nodes(graph = G, nodes = row[f'{dist-1}-children']),axis = 1)
        
        # display(df)
    
    step = f'{max_dist}-family acquired.'
    print(step)
    tqdm.pandas(desc=f'Calculating family      ')
    df['family'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-neighbors'] for dist in range(1, max_dist+1)]) - (row['source']), axis = 1)
    
    tqdm.pandas(desc=f'Calculating exclusion   ')
    df['family_without_children'] = df.progress_apply(lambda row: row['family'] -set.union(*[row[f'{dist}-children'] for dist in range(1, max_dist+1)]), axis = 1)


    # tqdm.pandas(desc=f'Calculating lineage          ')
    # df['lineage'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-children'] for dist in range(1, max_dist+1)]+
    #                                                          [row[f'{dist}-parents'] for dist in range(1, max_dist+1)] +
    #                                                          [row['source']]),axis=1)
    
    # tqdm.pandas(desc=f'Calculating family - lineage ')
    # df['family_without_lineage'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-neighbors'] for dist in range(1, max_dist+1)]) - row['lineage'],axis=1)
    # display(df)
    # Re-ordering columns :
    dists = range(1, max_dist+1)
    df = df[['source'] + [f'{dist}-neighbors' for dist in dists] + [f'{dist}-children' for dist in dists] + [f'{dist}-parents' for dist in dists]  + ['family', 'family_without_children']]



    return df

