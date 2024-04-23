from constants import *
from data_utils import load_node_matrix, load_edge_csv, get_nodelist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import os
import joblib

import optuna
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
import torch_geometric.transforms as T

  
def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func


# ----------------- Data processing ----------------- #
def load_athaliana_data(path, featureless=False):
    """
    Load Arabidopsis thaliana data from its file, preprocess it, and return it in a format suitable for graph-based machine learning models under the PyG framework.
    
    This function reads the CSV file, preprocesses the data, creates node lists and feature matrices for different types of nodes,
    and creates edge index arrays for different types of edges. It then packages all of this data into a HeteroData object and returns it.
    
    Parameters
    ----------
    path : str
        The path to the iric.csv file.

    Returns
    -------
    data : torch_geometric.data.HeteroData
        A HeteroData object containing the preprocessed data. This object includes node feature matrices for different types of nodes, 
        edge index arrays for different types of edges, and the number of nodes for each node type.
    """
    df = pd.read_csv(path, index_col=0, dtype=str)
    df['source_node'] = df.index
    df.columns = [x.lower() for x in df.columns]

    # List of all nodetypes
    nodelist = {
        "genes" : get_nodelist(df, 'genes'),
        "traito" : get_nodelist(df, 'traito'),
        "go" : get_nodelist(df, 'go'),
        "po" : get_nodelist(df, 'po'),
        "panther" : get_nodelist(df, 'panther'),
        "prosite_profiles" : get_nodelist(df, 'prosite_profiles'),
        "prosite_patterns" : get_nodelist(df, 'prosite_patterns'),
        "prints" : get_nodelist(df, 'prints'),
        "superfamily" : get_nodelist(df, 'superfamily')
    }

def load_iric_data(path, featureless=False):
    """
    Load iric data from its file, preprocess it, and return it in a format suitable for graph-based machine learning models under the PyG framework.

    This function reads the CSV file, preprocesses the data, creates node lists and feature matrices for different types of nodes, 
    and creates edge index arrays for different types of edges. It then packages all of this data into a HeteroData object and returns it.

    Parameters
    ----------
    path : str
        The path to the iric.csv file.

    Returns
    -------
    data : torch_geometric.data.HeteroData
        A HeteroData object containing the preprocessed data. This object includes node feature matrices for different types of nodes, 
        edge index arrays for different types of edges, and the number of nodes for each node type.
    """
    df = pd.read_csv(path, index_col=0, dtype=str)
    df['source_node'] = df.index
    df.columns = [x.lower() for x in df.columns]

    # List of all nodetypes
    nodelist = {
        "genes" : get_nodelist(df, 'genes'),
        "traito" : get_nodelist(df, 'traito'),
        "go" : get_nodelist(df, 'go'),
        "po" : get_nodelist(df, 'po'),
        "panther" : get_nodelist(df, 'panther'),
        "prosite_profiles" : get_nodelist(df, 'prosite_profiles'),
        "prosite_patterns" : get_nodelist(df, 'prosite_patterns'),
        "prints" : get_nodelist(df, 'prints'),
        "superfamily" : get_nodelist(df, 'superfamily')
    }

    if not featureless:
        features = {
            "genes": df[df.index.isin(nodelist["genes"])][IricGene.features.value],
            "traito": df[df.index.isin(nodelist["traito"])][IricTO.features.value],
            "go": df[df.index.isin(nodelist["go"])][IricGO.features.value],
            "po": df[df.index.isin(nodelist["po"])][IricPO.features.value],
            "panther": None,
            "prosite_profiles": None,
            "prosite_patterns": None,
            "prints": None,
            "superfamily": None,
        }

        genes_x, genes_mapping = load_node_matrix(
            nodelist['genes'], features['genes'], encoders={
                'biotype': OneHotEncoder(),
                'contig': OneHotEncoder(),
                # 'ncoils': PUEncoder(),
                # 'tmhmm': PUEncoder(),
                # 'annotation score' : TypeConverter(),
                'strand': TypeConverter(),
                'fmin': TypeConverter(),
                'fmax': TypeConverter(),
                # 'Genomic Sequence': GenomicSequenceEncoder(),
                # 'Protein Sequence': ProteinSequenceEncoder(),
                # 'InterPro:description': SentenceEncoder(),
                # 'Keywords': SentenceEncoder(),
                # 'Trait Class': SentenceEncoder(),
                # 'Allele': SentenceEncoder(),
                # 'Gene Name Synonyms': SentenceEncoder(),
                # 'Family': SentenceEncoder(),
                # Explanation: SentenceEncoder(),
            })

        go_x, go_mapping = load_node_matrix(
            nodelist['go'], features['go'], encoders={
                'namespace': OneHotEncoder(),
                # 'definition': SentenceEncoder(),
                # 'name': SentenceEncoder(),
            })

        po_x, po_mapping = load_node_matrix(
            nodelist['po'], features['po'], encoders={
                'namespace': OneHotEncoder(),
                # 'definition': SentenceEncoder(),
                # 'name': SentenceEncoder(),
            })

    else:
        features = {
            "genes": None,
            "traito": None,
            "go": None,
            "po": None,
            "panther": None,
            "prosite_profiles": None,
            "prosite_patterns": None,
            "prints": None,
            "superfamily": None,
        }

        _, genes_mapping = load_node_matrix(nodelist['genes'], features['genes'])
        _, go_mapping = load_node_matrix(nodelist['go'], features['go'])
        _, po_mapping = load_node_matrix(nodelist['po'], features['po'])


    # TODO : encoders for to

    _, to_mapping = load_node_matrix(nodelist['traito'], features['traito'])
    _, prosite_profiles_mapping = load_node_matrix(nodelist['prosite_profiles'], features['prosite_profiles'])
    _, prosite_patterns_mapping = load_node_matrix(nodelist['prosite_patterns'], features['prosite_patterns'])
    _, superfamily_mapping = load_node_matrix(nodelist['superfamily'], features['superfamily'])
    _, panther_mapping = load_node_matrix(nodelist['panther'], features['panther'])
    _, prints_mapping = load_node_matrix(nodelist['prints'], features['prints'])

    # Isolate columns representing links between nodes
    df_links = df[IricNode.links.value]
    df_links['source_node'] = df_links.index
        
    gene_gene_edge_index, _ = load_edge_csv(df_links[['source_node', 'interacts_with']].dropna(), 'source_node', genes_mapping, 'interacts_with', genes_mapping)
    gene_go_edge_index, _ = load_edge_csv(df_links[['source_node', 'gene ontology']].dropna(), 'source_node', genes_mapping, 'gene ontology', go_mapping)
    gene_to_edge_index, _ = load_edge_csv(df_links[['source_node', 'trait ontology']].dropna(), 'source_node', genes_mapping, 'trait ontology', to_mapping)
    gene_po_edge_index, _ = load_edge_csv(df_links[['source_node', 'plant ontology']].dropna(), 'source_node', genes_mapping, 'plant ontology', po_mapping)
    gene_prosite_profiles_edge_index, _ = load_edge_csv(df_links[['source_node', 'prosite_profiles']].dropna(), 'source_node', genes_mapping, 'prosite_profiles', prosite_profiles_mapping)
    gene_prosite_patterns_edge_index, _ = load_edge_csv(df_links[['source_node', 'prosite_patterns']].dropna(), 'source_node', genes_mapping, 'prosite_patterns', prosite_patterns_mapping)
    gene_superfamily_edge_index, _ = load_edge_csv(df_links[['source_node', 'superfamily']].dropna(), 'source_node', genes_mapping, 'superfamily', superfamily_mapping)
    gene_panther_edge_index, _ = load_edge_csv(df_links[['source_node', 'panther']].dropna(), 'source_node', genes_mapping, 'panther', panther_mapping)
    gene_prints_edge_index, _ = load_edge_csv(df_links[['source_node', 'prints']].dropna(), 'source_node', genes_mapping, 'prints', prints_mapping)
    go_go_edge_index, _ = load_edge_csv(df_links[df_links.source_node.str.match(r'(GO:\d+)')][['source_node', 'is_a']].dropna(), 'source_node', go_mapping, 'is_a', go_mapping)
    po_po_edge_index, _ = load_edge_csv(df_links[df_links.source_node.str.match(r'(PO:\d+)')][['source_node', 'is_a']].dropna(), 'source_node', po_mapping, 'is_a', po_mapping)
    to_to_edge_index, _ = load_edge_csv(df_links[df_links.source_node.str.match(r'(TO:\d+)')][['source_node', 'is_a']].dropna(), 'source_node', to_mapping, 'is_a', to_mapping)

    data = HeteroData()

    if not featureless:
        # Nodes with features
        data['genes'].x = genes_x  
        data['go'].x = go_x 
        data['po'].x = po_x
        data['traito'].x =  torch.ones(len(nodelist['traito']), 1)
        data['prosite_profiles'].x = torch.ones(len(nodelist['prosite_profiles']), 1)
        data['prosite_patterns'].x = torch.ones(len(nodelist['prosite_patterns']), 1)
        data['superfamily'].x = torch.ones(len(nodelist['superfamily']), 1)
        data['panther'].x = torch.ones(len(nodelist['panther']), 1)
        data['prints'].x = torch.ones(len(nodelist['prints']), 1)

    else:
        # Featureless nodes
        data['genes'].x = torch.ones(len(nodelist['genes']), 1)
        data['go'].x = torch.ones(len(nodelist['go']), 1)
        data['po'].x = torch.ones(len(nodelist['po']), 1)
        data['traito'].x =  torch.ones(len(nodelist['traito']), 1)
        data['prosite_profiles'].x = torch.ones(len(nodelist['prosite_profiles']), 1)
        data['prosite_patterns'].x = torch.ones(len(nodelist['prosite_patterns']), 1)
        data['superfamily'].x = torch.ones(len(nodelist['superfamily']), 1)
        data['panther'].x = torch.ones(len(nodelist['panther']), 1)
        data['prints'].x = torch.ones(len(nodelist['prints']), 1)

    # Links
    data['genes', 'interacts_with', 'genes'].edge_index = gene_gene_edge_index
    data['genes', 'gene_ontology', 'go'].edge_index = gene_go_edge_index
    data['genes', 'trait_ontology', 'traito'].edge_index = gene_to_edge_index
    data['genes', 'plant_ontology', 'po'].edge_index = gene_po_edge_index
    data['genes', 'profile', 'prosite_profiles'].edge_index = gene_prosite_profiles_edge_index
    data['genes', 'pattern', 'prosite_patterns'].edge_index = gene_prosite_patterns_edge_index
    data['genes', 'family', 'superfamily'].edge_index = gene_superfamily_edge_index
    data['genes', 'panther_id', 'panther'].edge_index = gene_panther_edge_index
    data['genes', 'prints_id', 'prints'].edge_index = gene_prints_edge_index
    data['go', 'is_a', 'go'].edge_index = go_go_edge_index
    data['po', 'is_a', 'po'].edge_index = po_po_edge_index
    data['traito', 'is_a', 'traito'].edge_index = to_to_edge_index

    return data

@timer_func
def split_data(data, config):
    """
    Split the data into training, validation, and test sets using a random link split.

    The split is performed on the edges of the graph, and the function ensures that the reverse edge types are split accordingly to avoid leakage.
    See the documentation for torch_geometric.loader.RandomLinkSplit for more information.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        The data to be split. This should be a HeteroData object containing the preprocessed data.
    config : object
        An object containing the configuration parameters for the split. This should include the ratios for the validation and test sets, 
        the disjoint train ratio, and the neg_sampling_ratio.

    Returns
    -------
    train_data, val_data, test_data : tuple of torch_geometric.data.HeteroData
        A tuple containing the training, validation, and test data. Each of these is a HeteroData object.
    """
    if config.labels:
        labels = (config.labels['head'], config.labels['relation'], config.labels['tail'])
        split = T.RandomLinkSplit(
            num_val= config.val_ratio,
            num_test= config.test_ratio,
            is_undirected=True,
            disjoint_train_ratio=config.disjoint_train_ratio, # Amount of supervision edges in train set
            neg_sampling_ratio=0,
            add_negative_train_samples=False, # If True, neg_sampling_ratio random negative samples per positive sample to the training set
            edge_types=[labels], # Only use these edge types for splitting
            rev_edge_types=[(config.labels['tail'], "rev_" + config.labels['relation'], config.labels['head'])], # Ensure that the reverse edge types are split accordingly to avoid leakage
        )
    else:
        split = T.RandomLinkSplit(
            num_val= config.val_ratio,
            num_test= config.test_ratio,
            is_undirected=True,
            disjoint_train_ratio=config.disjoint_train_ratio, # Amount of supervision edges in train set
            neg_sampling_ratio=0,
            add_negative_train_samples=False, # If True, neg_sampling_ratio random negative samples per positive sample to the training set
        )


    return split(data)

@timer_func
def build_dataloaders(train_data, val_data, test_data, config):
    """
    Build data loaders for training, validation, and testing.

    This function creates data loaders for the training, validation, and test sets. 
    The data loaders are used to load the data in batches during training and evaluation.
    Note that the loaders create their own negative samples.
    The validation and test loaders are created with a fixed number of negative samples, according to OGB's evaluation protocol.

    Parameters
    ----------
    train_data : torch_geometric.data.HeteroData
        The training data.
    val_data : torch_geometric.data.HeteroData
        The validation data.
    test_data : torch_geometric.data.HeteroData
        The test data.
    config : object
        An object containing the configuration parameters for the data loaders. This should include the number of neighbors to sample, 
        the negative sampling ratio, and the batch size.

    Returns
    -------
    train_loader, val_loader, test_loader : tuple of torch_geometric.loader.LinkNeighborLoader
        A tuple containing the data loaders for the training, validation, and test sets.
    """
    if config.labels:
        edge_label_index = ((config.labels['head'], config.labels['relation'], config.labels['tail']), train_data[config.labels['head'], config.labels['relation'], config.labels['tail']].edge_label_index)
    else:
        edge_label_index = None
        
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=config.num_neighbors, # Sample for each edge 20 neighbors in the first hop and at most 10 in the second.
        neg_sampling=NegativeSampling(mode="triplet", amount=config.train_neg_sampling_ratio), # Samples 500 negative edges per positive edge in the subgraph
        # Seed supervision edges from which to sample the subgraph that will be used for message passing.
        edge_label_index=edge_label_index,
        edge_label=None,
        batch_size=config.batch_size,
        shuffle=True, # Shuffle data each epoch
        pin_memory=True,
        num_workers=5
    )

    eval_batch_size = config.batch_size
    if config.train_neg_sampling_ratio < 1000: # Possible OOM issues as evaluation batch size are going to be larger than during training
        # Dynamically adapt batch size. This will cause evaluation to be slower, but it will prevent OOM errors.
        eval_batch_size = int(config.batch_size * (config.train_neg_sampling_ratio / 1000))

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=config.num_neighbors,
        neg_sampling=NegativeSampling(mode="triplet", amount=config.scorelist_size), # Samples 1000 negative edges per positive edge in the subgraph
        # Seed supervision edges from which to sample the subgraph that will be used for message passing.
        edge_label_index=edge_label_index,
        edge_label=None, # edge_label needs to be None for "triplet" negative sampling mode
        batch_size=eval_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=5
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=config.num_neighbors,
        neg_sampling=NegativeSampling(mode="triplet", amount=config.scorelist_size), # Samples 500 negative edges per positive edge in the subgraph
        # Seed supervision edges from which to sample the subgraph that will be used for message passing.
        edge_label_index=edge_label_index,
        edge_label=None, # edge_label needs to be None for "triplet" negative sampling mode
        batch_size=eval_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=5
    )

    return train_loader, val_loader, test_loader

@timer_func
def find_batch_size(model, train_data, criterion, optimizer, config, device):
    """
    Find the maximum batch size that can be used without running out of memory.

    This function iteratively increases the batch size until an out of memory (OOM) error occurs, 
    then returns the largest batch size that did not cause an OOM error. 
    It also returns the estimated duration of an epoch, which is used to skip the training of the model if training time is deemed too long.

    Parameters
    ---------- 
    model : torch.nn.Module
        The model to be trained.
    train_data : torch_geometric.data.HeteroData
        The training data.
    criterion : function
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    config : object
        An object containing the configuration parameters for the data loader. This should include the number of neighbors to sample, 
        and the negative sampling ratio.
    device : torch.device
        The device on which to perform computations.

    Returns
    -------
    batch_size : int
        The maximum batch size that can be used without running out of memory.
    epoch_duration : float
        The estimated duration of an epoch in seconds.

    Raises
    ------
    RuntimeError
        If an error other than an out of memory error occurs.
    """
    batch_size = 128
    epoch_duration = {}
    edge_label_index = ("genes", "gene_ontology", "go")
    while True:
        data_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=config.num_neighbors,
            neg_sampling=NegativeSampling(mode="triplet", amount=config.train_neg_sampling_ratio), # Samples 500 negative edges per positive edge in the subgraph
            edge_label_index=(edge_label_index, train_data[config.labels['head'], config.labels['relation'], config.labels['tail']].edge_label_index), # Seed supervision edges from which to sample the subgraph that will be used for training
            edge_label=None,
            batch_size=batch_size,
            shuffle=True, # Shuffle data each epoch
            pin_memory=True,
            num_workers=5
        )

        try:
            torch.cuda.synchronize() # Wait for all kernels in all streams on a CUDA device to complete
            start_time = time()
            sampled_data = next(iter(data_loader)).to(device)
            pred = model(sampled_data, config)
            pos_samples = torch.ones(len(sampled_data[config.labels['tail']].dst_pos_index), device=device)
            neg_samples = torch.zeros(len(sampled_data[config.labels['tail']].dst_neg_index.view(-1)), device=device) # As many zeroes as there are negative samples * batch_size
            ground_truth = torch.cat((pos_samples, neg_samples))

            loss = criterion(pred, ground_truth)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Batch size {batch_size} successful")

            # Case: all supervision edges can fit in memory at once
            if batch_size >= data_loader.edge_label_index[1].shape[1]:
                print(f"Since all supervision edges can fit in memory at once, setting batch_size to len of sup. edges. = {data_loader.edge_label_index[1].shape[1]}")
                torch.cuda.synchronize()
                epoch_duration[batch_size] = (time() - start_time) * len(data_loader)
                len_sup_edges = data_loader.edge_label_index[1].shape[1]
                try: 
                    del model
                    del data_loader
                    del sampled_data
                    del pred
                    del pos_samples
                    del neg_samples
                    del ground_truth
                    del loss
                except:
                    print('Error during memory freeup')
                    torch.cuda.empty_cache()
                torch.cuda.empty_cache()

                estimated_epoch_duration = epoch_duration[batch_size] * (len_sup_edges // batch_size)  # Approximate epoch duration
                print(f'Estimated epoch duration: {estimated_epoch_duration}s')

                return batch_size, estimated_epoch_duration
            
            # Case: OOM error did not occur, retry with double the batch size
            torch.cuda.synchronize()
            epoch_duration[batch_size] = (time() - start_time) * len(data_loader)
            batch_size *= 2

        except RuntimeError as e:
            # Case: OOM error

            if 'out of memory' in str(e):
                if batch_size == 128:
                    
                    print(f"Default batch size ({batch_size}) caused OOM error. Pruning trial..")
                    raise Pruning()
                
                print(f"Batch size {batch_size} caused OOM error, setting batch_size to {batch_size // 2}. Epoch duration estimated at {epoch_duration[batch_size // 2]}s.")
                # free up memory
                try:
                    del model
                    del data_loader
                    del sampled_data
                    del pred
                    del pos_samples
                    del neg_samples
                    del ground_truth
                    del loss
                except:
                    print('Error during memory freeup')
                torch.cuda.empty_cache()

                return batch_size // 2, epoch_duration[batch_size // 2]
            
            else: # Case: Unexpected error. Calls back to ExceptionHandler and fails the trial
                raise e
            
def get_sampled_edge_idx(sampled_data, config):
    """
    Get the indices of the sampled edges.

    This function extracts the indices of the positive and negative sampled edges from the sampled data. 
    It then concatenates these indices into two tensors: one for the source nodes and one for the destination nodes.

    Parameters
    ----------
    sampled_data : torch_geometric.data.Heterodata
        The sampled data. This should be a Heterodata object that includes the indices of the source nodes and the destination nodes 
        for the positive and negative sampled edges.

    Returns
    -------
    src_idx, dst_idx : tuple of torch.Tensor
        A tuple containing two tensors. The first tensor contains the indices of the source nodes, and the second tensor contains 
        the indices of the destination nodes.

    Raises
    ------
    ValueError
        If the data cannot be processed correctly.
    """
    for nodetype in sampled_data.keys():
        if sampled_data[nodetype].src:
            src_pos = sampled_data[config.labels['head']].src_index
        if sampled_data[nodetype].dst_pos:
            dst_pos = sampled_data[config.labels['tail']].dst_pos_index
            dst_neg = sampled_data[config.labels['tail']].dst_neg_index

    if dst_neg.dim() == 1: # If there is only one negative sample per positive sample
        src_idx = torch.cat((src_pos, src_pos))
        dst_idx = torch.cat((dst_pos, dst_neg))
    else:
        # Repeat source node indexes as many times as there are neg samples to match the length of the destination node indexes
        src_idx = torch.cat((src_pos, src_pos.repeat_interleave(dst_neg.shape[1])))
        dst_idx = torch.cat((dst_pos, dst_neg.view(-1)))

    return src_idx, dst_idx

def get_sampled_edge_indices(sampled_data, config):
    """
    Get the indices of the sampled edges.

    This function extracts the indices of the positive and negative sampled edges from the sampled data. 
    It then concatenates these indices into two tensors: one for the source nodes and one for the destination nodes.

    Parameters
    ----------
    sampled_data : torch_geometric.data.Heterodata
        The sampled data. This should be a Heterodata object that includes the indices of the source nodes and the destination nodes 
        for the positive and negative sampled edges.

    Returns
    -------
    src_idx, dst_idx : tuple of torch.Tensor
        A tuple containing two tensors. The first tensor contains the indices of the source nodes, and the second tensor contains 
        the indices of the destination nodes.

    Raises
    ------
    ValueError
        If the data cannot be processed correctly.
    """
    if config.homogeneous:
        dst_pos = sampled_data.dst_pos_index
        dst_neg = sampled_data.dst_neg_index
        src_pos = sampled_data.src_index
    else:
        dst_pos = sampled_data[config.labels['tail']].dst_pos_index
        dst_neg = sampled_data[config.labels['tail']].dst_neg_index
        src_pos = sampled_data[config.labels['head']].src_index

    if dst_neg.dim() == 1: # If there is only one negative sample per positive sample
        src_idx = torch.cat((src_pos, src_pos))
        dst_idx = torch.cat((dst_pos, dst_neg))
    else:
        # Repeat source node indexes as many times as there are neg samples to match the length of the destination node indexes
        src_idx = torch.cat((src_pos, src_pos.repeat_interleave(dst_neg.shape[1])))
        dst_idx = torch.cat((dst_pos, dst_neg.view(-1)))

    return src_idx, dst_idx

def get_sampled_nodes_idx(sampled_data, config):
    """
    Returns the node idx of the sampled triples used for eval.
    """
    src_emb_idx, dst_emb_idx = get_sampled_edge_indices(sampled_data)
    src_n_idx = sampled_data[config.labels['head']].n_id[src_emb_idx]
    dst_n_idx = sampled_data[config.labels['tail']].n_id[dst_emb_idx]
    return src_n_idx, dst_n_idx

# ----------------- Loss functions ----------------- #
def sigmoid_focal_loss( 
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

# ------------- Data Encoders ----------------- #

# class SequenceEncoder:
#     def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
#         self.device = device
#         self.model = SentenceTransformer(model_name, device=device)

#     @torch.no_grad()
#     def __call__(self, df):
#         x = self.model.encode(df.values, show_progress_bar=True,
#                               convert_to_tensor=True, device=self.device)
#         return x.cpu()


    
class OneHotEncoder:
    """
    Perform one-hot encoding on the input DataFrame.

    This method creates a tensor where each row corresponds to a data point and each column to a unique category. 
    The tensor is populated with 1s based on a mapping from categories to their corresponding index.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be encoded.
    nodelist : list
        A list of nodes.

    Returns
    -------
    x : torch.Tensor
        A tensor representing the one-hot encoded data.
    """
    def __init__(self):
        pass

    def __call__(self, df, nodelist):        
        # Extract unique categories from the input DataFrame
        categories = df.unique()
        
        # Create a mapping from categories to their corresponding index
        mapping = {categories: i for i, categories in enumerate(categories)}
        
        # Initialize a tensor with zeros, where rows correspond to data points and columns to unique categories
        x = torch.zeros(len(nodelist), len(mapping))

        # Populate the tensor with 1s based on the mapping
        for i, category in enumerate(df):
            x[i, mapping[category]] = 1
        
        return x
    
class TypeConverter:
    """
    A class used to convert strings in the DataFrame.

    This class provides a callable object that can be used to convert the data type of a pandas DataFrame's column to float. 
    The converted data is then converted to a PyTorch tensor.

    Methods
    -------
    __call__(df, nodelist):
        Convert the data type of the input DataFrame.

    """
    def __init__(self):
        pass

    def __call__(self, df, nodelist):
        # Convert df.values to np float
        x = df.values.astype(float)
        x = torch.tensor(x).unsqueeze(-1)

        return x

class PUEncoder:
    """
    A class used to encode non-NaN values in a DataFrame.

    This class provides a callable object that can be used to encode non-NaN values in a pandas DataFrame to 1, 
    while keeping NaN values as NaN. The encoded data is then converted to a PyTorch tensor.
    """
    def __init__(self):
        pass

    def __call__(self, df, nodelist):

        # Convert non-nan values to 1, and keep nan values as nan
        x = np.where(df.notnull(), 1, np.nan)
        x = torch.tensor(x).unsqueeze(-1)
        return x

# ----------------- Early stopping ----------------- #    
class EarlyStopper:
    """
    A class used to implement early stopping during model training.

    This class provides a callable object that can be used to monitor a score (in Link Prediction tasks, usually MRR) during model training. 
    If the score does not improve by a certain relative amount within a certain number of evaluations, the training is stopped early.

    Attributes
    ----------
    frequency : int
        The number of epochs between each evaluation of the score.
    patience : int
        The number of evaluations to wait for improvement before stopping.
    direction : str
        The direction of improvement ('minimize' or 'maximize').
    relative_delta : float
        The relative improvement threshold.
    counter : int
        A counter for the number of evaluations without improvement.
    best_score : float
        The best score observed so far.
    early_stop : bool
        A flag indicating whether to stop the training early.
    scores : list of float
        A list of the scores observed so far.

    Methods
    -------
    __call__(score):
        Update the early stopping status based on the latest score.
    is_best(score):
        Check whether the latest score is the best so far.

    Raises
    ------
    ValueError
        If the direction is not 'minimize' or 'maximize'.
    """
    def __init__(self, frequency=10, patience=2, direction='minimize', relative_delta=0.01):
        self.frequency = frequency # Evaluate every n epochs
        self.patience = patience # Number of evaluations to wait for improvement
        self.direction = direction # Direction of improvement
        self.relative_delta = relative_delta # Relative improvement threshold
        self.counter = 0 
        self.best_score = None
        self.early_stop = False
        self.scores = []

    def __call__(self, score):
        self.scores.append(score)
        if len(self.scores) % self.frequency == 0:  # Check scores at the specified frequency
            if self.is_best(score):
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                    self.early_stop = True

    def is_best(self, score):
        if self.direction == "minimize":
            if self.best_score is None or score < self.best_score * (1 - self.relative_delta):
                return True
        elif self.direction == "maximize":
            if self.best_score is None or score > self.best_score * (1 + self.relative_delta):
                return True
        return False
    
# --------------- Optuna Utils ---------------- # 
class OutOfTime(Exception):
    pass    

class ExceptionHandler(Exception):
    pass

class Pruning(Exception):
    pass

class DebugCallback():
    def __init__(self):
        pass

    def __call__(self, study, trial):
        print(f"Trial {trial.number} finished with status {trial.state}")
        pruned_trials = [t for t in study.trials if t.state == optuna.trial._state.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial._state.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial._state.TrialState.FAIL]
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of trials failed or skipped: ", len(failed_trials))
        print("  Number of complete trials: ", len(complete_trials))


class DataManager():
    """This class checks if the current trial is the best in the optuna study, and if it is not, it removes the saved model alongside the dataloaders"""
    def __init__(self):
        pass

    def __call__(self, study, trial):
        try:
            # Log study info and save the best model
            results_path = f"data/results/optuna"
            run_path = f"{results_path}/{study.study_name}/best_trial"
            os.makedirs(run_path, exist_ok=True)

            if study.best_trial is not None or trial.number == 0:
                if trial == study.best_trial or trial.number == 0:
                    print(f"Trial {trial.number} is the best, keeping saved model and dataloaders")
                    for file in os.listdir(run_path):
                        os.remove(f"{run_path}/{file}")
                    os.rename(f"{results_path}/{trial.number}_model.pt", f"{run_path}/{trial.number}_model.pt")
                    os.rename(f"{results_path}/{trial.number}_train_loader.pt", f"{run_path}/{trial.number}_train_loader.pt")
                    os.rename(f"{results_path}/{trial.number}_val_loader.pt", f"{run_path}/{trial.number}_val_loader.pt")
                    os.rename(f"{results_path}/{trial.number}_test_loader.pt", f"{run_path}/{trial.number}_test_loader.pt")
                    os.rename(f"{results_path}/{trial.number}_config.pt", f"{run_path}/{trial.number}_config.pt")
                    os.rename(f"{results_path}/{trial.number}_metadata.pt", f"{run_path}/{trial.number}_metadata.pt")

                    
                else:
                    print(f"Trial {trial.number} is not the best, removing saved model and dataloaders")
                    for file in os.listdir(results_path):
                        if not os.path.isdir(f"{results_path}/{file}"):
                            os.remove(f"{results_path}/{file}")
                joblib.dump(study, f"data/results/optuna/{study.study_name}/study_info_step_{trial.number}.pkl") # Save the study
        except:
            print("Error in DataManager for trial", trial.number)

# ----------------- Debugging ----------------- #
@timer_func
def heatmaps(preds, ground_truths, indexes):
    unique_indexes = torch.unique(indexes)

    for i, index in enumerate(unique_indexes):
        # Get the preds and ground_truths for the current index
        preds_i = preds[indexes == index]
        ground_truths_i = ground_truths[indexes == index]

        # Create a 2D array with ground_truths and diff
        data = torch.stack([ground_truths_i, preds_i], dim=1)

        # sort data by values in preds
        _, indices = torch.sort(data[:, 1], descending=True)
        data = data[indices]

        # Convert the tensors to numpy arrays for plotting
        data = data.cpu().numpy()

        # Create the heatmap
        plt.figure(figsize=(15, 10))
        plt.imshow(data, cmap='plasma', interpolation='nearest', aspect=0.008)
        plt.colorbar()
        plt.title(f"Heatmap for index {index.item()}")
        plt.xticks([0, 1], ['Ground Truth', 'Pred value'])

        # Log the heatmap to wandb
        # Close the figure to free up memory
        break
    return plt
