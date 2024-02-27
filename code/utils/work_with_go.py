from goatools import obo_parser
import torch
import warnings

def load_go_file(path_to_go_file):
    """
    Load an obo file for goatools.

    Parameters:
    - path_to_go_file (str): Filepath to the .obo file.
    
    Returns:
    - loaded_go (goatools.obo_parser.GODag) : The ontology loaded from the .obo input file, ready to be used with goatools.
    """
    loaded_go = obo_parser.GODag(path_to_go_file)
    return loaded_go

def node_depth_in_go(node_id, ontology, mute_warning = True):
    '''
    Returns the length of the shortest path from a node to the root of a given ontology.
    /!\ IF NODE_ID NOT FOUND in the ontology, return -1 and raise a warning.

    Parameters:
    - ontology (dict): Dictionary representing the ontology where keys are node IDs and values contain node information.
    - node_id (str): The node GO identifier. Example: 'GO:0048527'

    Returns:
    - depth (int): Length of the shortest path from a node to the root of a given ontology.
      '''

    try:
        depth = int(ontology[node_id].level)
    except KeyError:
        depth = -1
        if mute_warning == False:
          warnings.warn(f"Node ID '{node_id}' not found in the ontology.")
        
    return depth


def nodes_depths_in_go(nodes_ids, ontology):
    '''
    Returns a torch.tensor containing the depths of the input nodes.
    Depth is the length of the shortest path from a node to the root of a given ontology.
    /!\ If a node is NOT FOUND in the ontology, return -1 and raise a warning.
    
    Parameters:
    - nodes_ids (list) : The node GO identifiers. Example : ['GO:0048527', 'GO:0048528']
    - ontology (goatools.obo_parser.GODag) : An ontology that can be loaded with obo_parser.GODag(path_to_go_file).
                                             /!\ Some GO files are NOT granted without cycles, using it could create problems.
    Returns:
    - typed_depths (torch.Tensor) : Lengths of the shortest paths from nodes to the root of the given ontology.
                                    typed_depths.size() = torch.Size([len(nodes_ids)])
      '''
    depths = []
    for node_id in nodes_ids:
        depth = node_depth_in_go(node_id, ontology)
        depths.append(depth)
    typed_depths = torch.Tensor(depths)
    return typed_depths