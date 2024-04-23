print("\nImports...")
import sys
sys.path.append( '/home/ebutz/ESL2024/code/utils/train_utils' )

from play_with_complex import *
from train_utils import *
from model_utils import *
from ogb_utils import *

import warnings
import tqdm
import os
import wandb
import torch
from lion_pytorch import Lion
import torch_geometric.transforms as T
from torch_geometric.nn import *
from torch_geometric.nn.conv import *
from torch_geometric.nn.models import *
from torch_geometric.nn.norm import *
from torchmetrics.retrieval import RetrievalMRR, RetrievalHitRate


warnings.filterwarnings('ignore')

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

# ----------------- Config ----------------- #
run = wandb.init(project='these', mode='online', name='pyg_iric_noisy_labels_0_1_sig_full_intermediary_scorelist1000')
config = wandb.config

# Train parameters
config.homogeneous = False
config.labels = {'head' : 'genes', 'relation' : 'gene_ontology', 'tail' : 'go'}

# config.labels = {'head' : 'protein', 'relation' : '43', 'tail' : 'function'}
config.scorelist_size = 1000 # Number of negative labels to rank per positive label on validation and test sets. This should not be changed as metrics depend on this value.
config.split_ratio = 0.8
config.val_ratio = 0.1
config.test_ratio = 0.1
config.num_neighbors = [70, 55, 13, 89, 85]
config.batch_size = 1024
config.train_neg_sampling_ratio = 224 # 224
config.epochs = 18
config.disjoint_train_ratio = 0.6
config.lr = 0.0015308253347932983

# Early stopping parameters
config.stopper_metric = "mrr"
config.stopper_direction = "maximize" # Whether the goal is to "minimize" or "maximize" the metric
config.stopper_patience = 5
config.stopper_frequency = 1
config.stopper_relative_delta = 0.05

# Loss parameters
config.gamma = 1.3
config.alpha = 0.42680473078813763 

# Model parameters
config.gnn_layer = "ResGatedGraphConv"
config.dropout = 0.1
config.norm = 'DiffGroupNorm'
config.aggregation = 'min'
config.hidden_channels = 115
config.num_layers = 3
config.attention_heads = 4

print(config)

# ----------------- Data loading ----------------- #
# train_data, val_data, test_data = load_ogb('ogbl-biokg')

# data = train_data.update(val_data).update(test_data)
# data  = T.ToUndirected(merge=True)(data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
# data = T.RemoveDuplicatedEdges()(data) # Remove duplicated edges


# data = data.to_homogeneous()

# train_data  = T.ToUndirected(merge=True)(train_data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
# train_data = T.RemoveDuplicatedEdges()(train_data) # Remove duplicated edges
# # train_data  = train_data.to_homogeneous()

# val_data  = T.ToUndirected(merge=True)(val_data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
# val_data = T.RemoveDuplicatedEdges()(val_data) # Remove duplicated edges
# # val_data = val_data.to_homogeneous()

# test_data  = T.ToUndirected(merge=True)(test_data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
# test_data = T.RemoveDuplicatedEdges()(test_data) # Remove duplicated edges
# test_data = test_data.to_homogeneous()

# # Intialize node features in all data splits to be [0] for all nodes
# data.x = torch.zeros(data.num_nodes, 1)
# train_data.x = torch.zeros(train_data.num_nodes, 1)
# val_data.x = torch.zeros(val_data.num_nodes, 1)
# test_data.x = torch.zeros(test_data.num_nodes, 1)

# for edgetype in data.edge_types:
#     edgetype.x = torch.zeros(.num_nodes, 1)


# for nodetype in data.node_types:
#     data[nodetype].x = torch.ones(data[nodetype].num_nodes, 1)

# assert train_data.validate()
# assert val_data.validate()
# assert test_data.validate()
# assert data.validate()
# ----------------- Data loading ----------------- #
data = load_iric_data('data/iric.csv', featureless=False)
data  = T.ToUndirected(merge=True)(data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
data = T.RemoveDuplicatedEdges()(data) # Remove duplicated edges
print(data)
assert data.validate()

train_data, val_data, test_data = split_data(data, config)
train_loader, val_loader, test_loader = build_dataloaders(train_data, val_data, test_data, config)

# ----------------- Model ----------------- #
gnn_layers = get_gnn_layers(config)
norm_layers = get_norm_layers(config, len(data.node_types))

    
# ----------------- Loops ----------------- #
@timer_func
def evaluate(config, loader, model, criterion, compute_all_metrics=False, loader_type='validation', stopper_metric=False):
    """
    Evaluate the model on a given data loader.

    Parameters
    ----------
    config : object
        An object containing the configuration parameters for the model.    
    loader : DataLoader
        The data loader to evaluate the model on.
    model : Model
        The model to evaluate.
    criterion : callable
        The loss function to use for evaluation.
    compute_all_metrics : bool, optional
        Whether to compute all metrics or partially. Default is False.
    loader_type : str, optional
        The type of the loader ('validation' or 'test'). Default is 'validation'.
    stopper_metric : str or bool, optional
        The metric that dictates when to stop training. If False, only the loss is computed. Default is False.

    Returns
    -------
    tuple
        A tuple of the evaluation metric and the loss, depending on the value of `stopper_metric`.
        If compute_all_metrics is True, the function logs all metrics to W&B.
    """
    model.eval()
    num_neg_samples = loader.neg_sampling.amount
    with torch.no_grad():
        if stopper_metric or compute_all_metrics:
            ground_truths = torch.tensor([], device='cpu')
            preds = torch.tensor([], device='cpu')
            indexes = torch.tensor([], device='cpu')
            total_loss, total_examples = 0, 0
            index_end = loader.batch_size

        for sampled_data in loader:
            sampled_data = sampled_data.to(device)

            batch_size = len(sampled_data[config.labels['tail']].dst_pos_index)
            pred = model(sampled_data, config)
            pos_samples = torch.ones(batch_size, device=device)
            neg_samples = torch.zeros(num_neg_samples*batch_size, device=device)
            ground_truth = torch.cat((pos_samples, neg_samples))

            if stopper_metric or compute_all_metrics: # Store preds and truths for all batches, and compute indices to calc metrics
                index_pos = torch.arange(end=index_end, start=index_end-batch_size) # index for predictions with pos. ground_truth
                index_neg = torch.arange(end=index_end, start=index_end-batch_size).repeat_interleave(num_neg_samples)
                index = torch.cat((index_pos, index_neg))
                indexes = torch.cat((indexes, index.to('cpu')))
                preds = torch.cat((preds, pred.to('cpu')))
                ground_truths = torch.cat((ground_truths, ground_truth.to('cpu')))
                index_end += batch_size

            eval_loss = criterion(pred, ground_truth, gamma=config.gamma, alpha=config.alpha)

            # if not stopper_metric: # Just logging loss
            #     wandb.log({"running_val_loss": eval_loss})
            #     break
            # else:
            total_loss += float(eval_loss) * pred.numel() 
            total_examples += pred.numel()
            eval_loss = total_loss / total_examples

            # if stopper_metric and not compute_all_metrics:
            #     if index_end >= 2048: # Compute approx. intermediate metric on a few datapoints to speed up hyperopt process
            #         break 

        model.train()

        if stopper_metric and not compute_all_metrics: # Compute only the metric that dictates stopping
            indexes = indexes.long()

            # # Log heatmap between prediction and ground truth. Useful for visualization / debugging. overhead 0.2s per epoch for a single sample.
            # heatmap = heatmaps(preds, ground_truths, indexes) 
            # wandb.log({f"heatmap": wandb.Image(heatmap)})
            # heatmap.close() # Free up memory

            match stopper_metric:
                case "val_loss":
                    eval_loss = total_loss / total_examples
                    return eval_loss
                
                case "mrr":
                    mrr = RetrievalMRR().to(device)
                    return mrr(preds, ground_truths, indexes=indexes), eval_loss
                
                case "hit_at_10":
                    hit_at_10 = RetrievalHitRate(top_k=10).to(device)
                    return hit_at_10(preds, ground_truths, indexes=indexes), eval_loss
                
                case "hit_at_5":
                    hit_at_5 = RetrievalHitRate(top_k=5).to(device)
                    return hit_at_5(preds, ground_truths, indexes=indexes), eval_loss
                
                case "hit_at_3":
                    hit_at_3 = RetrievalHitRate(top_k=3).to(device)
                    return hit_at_3(preds, ground_truths, indexes=indexes), eval_loss
                
                case "hit_at_1":
                    hit_at_1 = RetrievalHitRate(top_k=1).to(device)
                    return hit_at_1(preds, ground_truths, indexes=indexes), eval_loss
                
                case _:
                    raise ValueError(f"Unrecognized stopper metric: '{stopper_metric}'")
                
        if compute_all_metrics: # Compute all metrics at the end of training
            indexes = indexes.long()
            mrr = RetrievalMRR().to(device)
            hit_at_10 = RetrievalHitRate(top_k=10).to(device)
            hit_at_5 = RetrievalHitRate(top_k=5).to(device)
            hit_at_3 = RetrievalHitRate(top_k=3).to(device)
            hit_at_1 = RetrievalHitRate(top_k=1).to(device)
            
            mrr = mrr(preds, ground_truths, indexes=indexes)
            hit_at_10 = hit_at_10(preds, ground_truths, indexes=indexes)
            hit_at_5 = hit_at_5(preds, ground_truths, indexes=indexes)
            hit_at_3 = hit_at_3(preds, ground_truths, indexes=indexes)
            hit_at_1 = hit_at_1(preds, ground_truths, indexes=indexes)

            wandb.log({
                f"{loader_type}MRR": mrr, f"{loader_type}hit_at_10": hit_at_10, f"{loader_type}hit_at_5": hit_at_5, f"{loader_type}hit_at_3": hit_at_3, f"{loader_type}hit_at_1": hit_at_1
            })


# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

model = Model(config, data, norm_layers, gnn_layers).to(device)
print(model)

criterion = sigmoid_focal_loss
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
# optimizer = Lion(model.parameters(), lr=config.lr, weight_decay=1e-2)
early_stopper = EarlyStopper(frequency=config.stopper_frequency, patience=config.stopper_patience, direction=config.stopper_direction, relative_delta=config.stopper_relative_delta)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

@timer_func
def train(config, train_loader, val_loader, model, criterion, optimizer, early_stopper):
    for epoch in range(config.epochs):
        total_loss = total_examples = 0

        for sampled_data in tqdm.tqdm(train_loader, desc="Training"):

            sampled_data = sampled_data.to(device)
            pred = model(sampled_data, config)
            pos_samples = torch.ones(len(sampled_data[config.labels['tail']].dst_pos_index), device=device)
            neg_samples = torch.zeros(len(sampled_data[config.labels['tail']].dst_neg_index.view(-1)), device=device) # As many zeroes as there are negative samples * batch_size
            ground_truth = torch.cat((pos_samples, neg_samples))
            # Add gaussian noise to pos_samples to simulate noisy labels. Added noise must be negative and not exceed 1.
            ground_truth += torch.normal(mean=0, std=1, size=(len(ground_truth),), device=device)
            # apply sigmoid
            ground_truth = torch.sigmoid(ground_truth)
            # ground_truth = torch.sigmoid(ground_truth)
            loss = criterion(pred, ground_truth, gamma=config.gamma, alpha=config.alpha)
            wandb.log({"loss": loss})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        train_loss = total_loss / total_examples
        print(f"Epoch: {epoch:03d}, Avg. Loss: {train_loss:.10f}")
        # scheduler.step()

        # Compute metrics and check for early stopping
        score, val_loss = evaluate(config, val_loader, model, criterion, stopper_metric=config.stopper_metric)
        wandb.log({"avg_loss": train_loss, "val_loss": val_loss, f"{config.stopper_metric}": score})

        # early_stopper(score)
        # if early_stopper.early_stop:
        #     print("Early stopping triggered at epoch", epoch)
        #     break

    print("Training done.")

train(config, train_loader, val_loader, model, criterion, optimizer, early_stopper)
evaluate(config, val_loader, model, criterion, compute_all_metrics=True, loader_type='validation', stopper_metric=False)
evaluate(config, test_loader, model, criterion, compute_all_metrics=True, loader_type='test', stopper_metric=False)