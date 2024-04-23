
import regex as re
import json
import pandas as pd
import torch

from constants import *

def remove_features_from_triples(triples, features_to_remove):
    """
    Remove features from a triples DataFrame

    Parameters:
    - triples (pandas.DataFrame): Input triples DataFrame
    - features_to_remove (list): List of features to remove

    Returns:
    - triples (pandas.DataFrame): Output triples DataFrame
    """
    for feature in features_to_remove:
        triples = triples[~triples['predicate'].str.match(feature)]
    return triples

def load_as_triples(filepath):

    # Load nested dictionary from a JSON file
    with open(filepath, 'r') as file:
        input_dict = json.load(file)
    subjects = []
    predicates = []
    objects = []

    # Iterate through the nested dictionary
    for subject, predicates_dict in input_dict.items():
        for predicate, object in predicates_dict.items():
            if type(object) != str or '|' not in object:
                subjects.append(subject)
                predicates.append(predicate)
                objects.append(object)
            else: # If there are multiple objects
                for obj in object.split('|'):
                    # Append data to lists
                    subjects.append(subject)
                    predicates.append(predicate)
                    objects.append(obj)

    # Create a Pandas DataFrame
    df = pd.DataFrame({'subject': subjects, 'predicate': predicates, 'object': objects})

    # Display the resulting DataFrame
    return df

def get_nodelist(df, node_type):
    nodelist = set()

    def process_node(series, regex_pattern):
        for val in series:
            if '|' in str(val) and re.match(regex_pattern, str(val)):
                for node in str(val).split('|'):
                    nodelist.add(node)
            else:
                if re.match(regex_pattern, str(val)):
                    nodelist.add(val)

    match node_type:
        case 'genes':
            # Use vectorized operations to improve performance
            source_nodes = df['source_node'].str.extract(r'(^OsNippo\S+)').dropna()
            nodelist.update(source_nodes.squeeze())
            
            # Split the 'interacts_with' column and add nodes to the set
            interacts_with_nodes = df['interacts_with'].str.split('|').explode().dropna()
            nodelist.update(interacts_with_nodes)

        case 'go':
            pattern = r'^GO:\d{7}'
            process_node(df['source_node'], pattern)
            process_node(df['gene ontology'], pattern)
            process_node(df['is_a'], pattern)

        case 'po':
            pattern = r'^PO:\d{7}'
            process_node(df['source_node'], pattern)
            process_node(df['plant ontology'], pattern)
            process_node(df['is_a'], pattern)

        case 'traito':
            pattern = r'^TO:\d{7}'
            process_node(df['source_node'], pattern)
            process_node(df['trait ontology'], pattern)
            process_node(df['is_a'], pattern)

        case 'prosite_profiles':
            pattern = r'^PS\d{5}'
            process_node(df['prosite_profiles'], pattern)

        case 'prosite_patterns':
            pattern = r'^PS\d{5}'
            process_node(df['prosite_patterns'], pattern)

        case 'superfamily':
            pattern = r'^SSF\d{5}'
            process_node(df['superfamily'], pattern)

        case 'panther':
            pattern = r'^PTHR\d{5}'
            process_node(df['panther'], pattern)
            # Remove in nodelist all nodes matching panther pattern that contain a ':' (avoids duplicates due to subfamilies)
            nodelist = {node for node in nodelist if  re.match(r'^[^:]*$', node)}

        case 'prints':
            pattern = r'^PR\d{5}'
            process_node(df['prints'], pattern)
        
        case _:
            nodelist.update(df[node_type].dropna().unique().tolist())

    return list(nodelist)


def get_nodelist_by_regex(triples, regex_pattern):
    """
    Returns a list of nodes that match the regex pattern in either the subject or object column.
    """
    nodelist = set()
    subject_nodes = triples[triples['subject'].str.match(regex_pattern)]

    triples['object'] = triples['object'].astype(str) # Can only assess regex on strings
    object_nodes = triples[triples['object'].str.match(regex_pattern)]

    nodelist.update(subject_nodes['subject'])
    nodelist.update(object_nodes['object'])
    return list(nodelist)

def load_node_matrix(nodelist, node_features, encoders=None, **kwargs):
    # reset index
    mapping = {index: i for i, index in enumerate(nodelist)}

    x = torch.tensor([])
    if encoders is not None:
        xs = [encoder(node_features[col], nodelist) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1).to(torch.float32)

    return x, mapping

def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    src, dst = [], []
    for index in df.index:
        for target_node in df.loc[index, dst_index_col].split('|'):
            src.append(src_mapping[index])
            dst.append(dst_mapping[target_node])
    # print(f'src: (name: {src_index_col}, count: {len(src)}), dst: (name: {dst_index_col}, count: {len(dst)})')
    # print(f'set src: {len(set(src))}, set dst: {len(set(dst))}')
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


def add_to_dict(node, feature_name, feature_value, dictionary):
    """
    Add a node and its features to a dictionary

    Parameters:
    - node (str): Node name
    - feature_name (str): Feature name
    - feature_value (str): Feature value
    - dictionary (dict): Dictionary to add the node to
    """

    if node not in dictionary:
        dictionary[node] = {}
        dictionary[node].update({feature_name: [feature_value]})
    else:
        dictionary[node][feature_name].append(feature_value)

    return dictionary

def convert_to_dict(dataset):
    """
    Convert a pandas dataframe into multiple node matrices and a link matrix in a dict format

    Parameters:
    - dataset (pandas.DataFrame): Input dataset in a triple format with columns 'subject', 'predicate' and 'object'

    Returns:
    - genes_features (dict): Dictionary of genes_features {node: {feature_name: feature_value}}
    - go (dict): Dictionary of Gene Ontology terms {node: {feature_name: feature_value}}
    - po (dict): Dictionary of Plant Ontology terms {node: {feature_name: feature_value}}
    - to (dict): Dictionary of Trait Ontology terms {node: {feature_name: feature_value}}
    - prosite_profiles (dict): Dictionary of Prosite profiles {node: {feature_name: feature_value}}
    - prosite_patterns (dict): Dictionary of Prosite patterns {node: {feature_name: feature_value}}
    - superfamily (dict): Dictionary of SuperFamily terms {node: {feature_name: feature_value}}
    - panther (dict): Dictionary of PANTHER terms {node: {feature_name: feature_value}}
    - prints (dict): Dictionary of PRINTS terms {node: {feature_name: feature_value}}
    - links (dict): Dictionary of links {node: {feature_name: feature_value}}
    """
    # Transform dataset to nested dictionaries like {node: {feature_name: feature_value}}
    data_dict = dataset.groupby('subject').apply(lambda group: group.set_index('predicate').to_dict()['object']).to_dict()

    # Create dictionaries for each node type, and for links
    gene_nodes, go_nodes, po_nodes, to_nodes = set(), set(), set(), set()
    genes_features, go_features, po_features, to_features = {}, {}, {}, {}
    prosite_profiles, prosite_patterns, superfamily, panther, prints = {}, {}, {}, {}, {}
    gene_links, go_links, po_links, to_links, prosite_profiles_links, prosite_patterns_links, superfamily_links, panther_links, prints_links = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for node, features in data_dict.items():
        for feature_name, feature_value  in features.items():

            # Handle Gene type
            if feature_name in ['contig', 'fmin', 'fmax', 'strand', 'Annotation score', 'TMHMM', 'ncoils', 'Genomic Sequence', 'Protein Sequence', 'biotype', 'description', 'InterPro:description', 'Keyword', 'Trait Class', 'Allele', 'Gene Name Synonyms', 'Family', 'Explanation']:
                add_to_dict(node, feature_name, feature_value, genes_features)
                gene_nodes.add(node)

            # Handle Ontology types
            elif feature_name in ['namespace', 'definition', 'name']:
                if re.match(r'^GO:\d+$', node):
                    if node not in go_features:
                        go_features[node] = {}
                    go_features[node].update({feature_name: feature_value})
                    go_nodes.add(node)

                elif re.match(r'^PO:\d+$', node):
                    if node not in po_features:
                        po_features[node] = {}
                    po_features[node].update({feature_name: feature_value})
                    po_nodes.add(node)

                elif re.match(r'^TO:\d+$', node):
                    if node not in to_features:
                        to_features[node] = {}
                    to_features[node].update({feature_name: feature_value})
                    to_nodes.add(node)

            # Handle ontology links
            elif feature_name == 'is_a':
                if re.match(r'^GO:\d+$', node):
                    add_to_dict(node, feature_name, feature_value, go_links)
                    go_nodes.add(node)

                elif re.match(r'^PO:\d+$', node):
                    add_to_dict(node, feature_name, feature_value, po_links)
                    po_nodes.add(node)

                elif re.match(r'^TO:\d+$', node):
                    add_to_dict(node, feature_name, feature_value, to_links)
                    to_nodes.add(node)
            

            # Handle featureless node
            elif feature_name == 'Prosite_profiles':
                add_to_dict(node, feature_name, feature_value, gene_links)
                add_to_dict(feature_value, feature_name, node, prosite_profiles_links) # Reverse links
                prosite_profiles[feature_value] = None # Featureless node
                            
            elif feature_name == 'Prosite_patterns':
                add_to_dict(node, feature_name, feature_value, gene_links)
                add_to_dict(feature_value, feature_name, node, prosite_patterns_links)
                prosite_patterns[feature_value] = None # Featureless node


            elif feature_name == 'SuperFamily':
                add_to_dict(node, feature_name, feature_value, gene_links)
                add_to_dict(feature_value, feature_name, node, superfamily_links)
                superfamily[feature_value] = None # Featureless node


            elif feature_name == 'PANTHER':
                add_to_dict(node, feature_name, feature_value, gene_links)
                add_to_dict(feature_value, feature_name, node, panther_links)
                panther[feature_value] = None # Featureless node
            
            elif feature_name == 'PRINTS':
                add_to_dict(node, feature_name, feature_value, gene_links)
                add_to_dict(feature_value, feature_name, node, prints_links)
                prints[feature_value] = None # Featureless node


            # Handles gene-gene links, functionnal annotation links (gene-ontologies)
            else:
                add_to_dict(node, feature_name, feature_value, gene_links)
                gene_nodes.add(node)
                match feature_name:
                    case 'Gene Ontology':
                        add_to_dict(feature_value, feature_name, node, go_links)
                    case 'Trait Ontology':
                        add_to_dict(feature_value, feature_name, node, to_links)
                    case 'Plant Ontology':
                        add_to_dict(feature_value, feature_name, node, po_links)
                    # interacts_with is already reversed 


    return gene_nodes, go_nodes, po_nodes, to_nodes, genes_features, go_features, po_features, to_features, prosite_profiles, prosite_patterns, superfamily, panther, prints, gene_links, go_links, po_links, to_links, prosite_profiles_links, prosite_patterns_links, superfamily_links, panther_links, prints_links

def dataframe_to_triple(df):
    """
    Convert a dataframe to a triple format

    Parameters:
    - df (pandas.DataFrame): Input dataframe

    Returns:
    - triples (pandas.DataFrame): Output dataframe in triple format
    """
    # Create a list of triples
    triples = []
    
    # Drop feature columns
    df.drop(columns=IricNode.features, inplace=True)
    for index, row in df.iterrows():
        for column in df.columns:
            for value in row[column].split('|'):
                triples.append([index, column, value])

    # Create a dataframe from the list of triples
    return pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])

if __name__ == '__main__':
    import pandas as pd

    dataset = pd.read_csv('data/iric_triples.csv')
    genes, go, po, to, prosite_profiles, prosite_patterns, superfamily, panther, prints, gene_links, go_links, po_links, to_links = convert_to_dict(dataset)