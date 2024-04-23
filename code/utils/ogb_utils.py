from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
import torch

def load_ogb(name):
    dataset = PygLinkPropPredDataset(name=name)

    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    num_nodes_dict = dataset.data.edge_stores[0]['num_nodes_dict']

    def ogbl_to_heterodata(edge_dict, num_nodes_dict):
        heterodata = HeteroData()

        for node_type in num_nodes_dict.keys():
            heterodata[node_type].x = torch.randn(num_nodes_dict[node_type], 0)

        from collections import defaultdict

        # Initialize a dictionary to hold lists of head_ids and tail_ids for each (head_type, edge_type, tail_type) tuple
        edge_index_dict = defaultdict(lambda: defaultdict(list))

        # Collect all the data
        for head_type, tail_type, edge_type, head_id, tail_id in zip(edge_dict['head_type'], edge_dict['tail_type'], edge_dict['relation'], edge_dict['head'], edge_dict['tail']):
            edge_type = str(edge_type.item())
            edge_index_dict[(head_type, edge_type, tail_type)]['head'].append(head_id)
            edge_index_dict[(head_type, edge_type, tail_type)]['tail'].append(tail_id)

        # Create the tensors
        for key, value in edge_index_dict.items():
            head_type, edge_type, tail_type = key
            head_ids = torch.tensor(value['head'], dtype=torch.long)
            tail_ids = torch.tensor(value['tail'], dtype=torch.long)
            heterodata[head_type, edge_type, tail_type].edge_index = torch.stack([head_ids, tail_ids], dim=0)
        return heterodata

    train_data = ogbl_to_heterodata(train_edge, num_nodes_dict)
    val_data = ogbl_to_heterodata(valid_edge, num_nodes_dict)
    test_data = ogbl_to_heterodata(test_edge, num_nodes_dict)

    print(train_data, val_data, test_data)

    return train_data, val_data, test_data

def build_ogbl_dataloaders(train_data, val_data, test_data, config):
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
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=config.num_neighbors, # Sample for each edge 20 neighbors in the first hop and at most 10 in the second.
        neg_sampling=NegativeSampling(mode="triplet", amount=config.train_neg_sampling_ratio), # Samples 500 negative edges per positive edge in the subgraph
        edge_label_index=None, # Seed supervision edges from which to sample the subgraph that will be used for message passing.
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
        neg_sampling=NegativeSampling(mode="triplet", amount=1000), # Samples 1000 negative edges per positive edge in the subgraph
        edge_label_index=None,
        edge_label=None, # edge_label needs to be None for "triplet" negative sampling mode
        batch_size=eval_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=5
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=config.num_neighbors,
        neg_sampling=NegativeSampling(mode="triplet", amount=1000), # Samples 500 negative edges per positive edge in the subgraph
        edge_label_index=None,
        edge_label=None, # edge_label needs to be None for "triplet" negative sampling mode
        batch_size=eval_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=5
    )

    return train_loader, val_loader, test_loader

# ==== Expected output format of Evaluator for ogbl-biokg
# {'hits@1_list': hits@1_list, 'hits@3_list': hits@3_list, 
# 'hits@10_list': hits@10_list, 'mrr_list': mrr_list}
# - mrr_list (list of float): list of scores for calculating MRR 
# - hits@1_list (list of float): list of scores for calculating Hits@1 
# - hits@3_list (list of float): list of scores to calculating Hits@3
# - hits@10_list (list of float): list of scores to calculating Hits@10
# Note: i-th element corresponds to the prediction score for the i-th edge.
# Note: To obtain the final score, you need to concatenate the lists of scores and take average over the concatenated list.
def ogbl_eval(name, y_pred_pos, y_pred_neg):
    """Evaluate the predictions on the ogbl-biokg dataset.
    
    Args:
        name (str): The name of the dataset.
        y_pred_pos (torch.Tensor): The predicted scores for positive edges.
        y_pred_neg (torch.Tensor): The predicted scores for negative edges.
        
    Returns:
        dict: The evaluation results
    """
    evaluator = Evaluator(name=name)
    results = evaluator.eval({
        'y_pred_pos': y_pred_pos,
        'y_pred_neg': y_pred_neg
    })
    # Cat list of scores and take average
    hits_1 = sum(results['hits@1_list']) / len(results['hits@1_list'])
    hits_3 = sum(results['hits@3_list']) / len(results['hits@3_list'])
    hits_10 = sum(results['hits@10_list']) / len(results['hits@10_list'])
    mrr = sum(results['mrr_list']) / len(results['mrr_list'])
    return {
        'hits@1': hits_1,
        'hits@3': hits_3,
        'hits@10': hits_10,
        'mrr': mrr
    }

