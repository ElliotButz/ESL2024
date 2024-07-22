import torch
import os
import sys
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

epochs = 1
# Iric
iric_path = '/home/ebutz/ESL2024/data/full_iric/iric.csv'
mapped_iric_path  = '/home/ebutz/ESL2024/data/full_iric/altailed_mapped_iric.pickle'
altails_dict_path = '/home/ebutz/ESL2024/data/full_iric/altail_iric_DICT.pickle'
check_dicts = True

ontology_path = '/home/ebutz/ESL2024/data/go-basic.json.gz'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: '{device}'")


# # ------------- Loading ontology ------------- #

print("\nLoading ontology...")
nxo = from_file(ontology_path)
nxo.freeze()
pwc.nxo = nxo

df = pd.read_csv(iric_path, index_col=0, dtype=str)
df['source_node'] = df.index
df.columns = [x.lower() for x in df.columns]
go_map = get_nodelist(df, 'go')
go_to_idx = {node: i for i, node in enumerate(go_map)}
idx_to_go = {i: node for i, node in enumerate(go_map)}

# # ------------- Making global variables accessibles to pwc ------------- #

pwc.map_to_GO        = idx_to_go
pwc.mapped_alt_tails = None
pwc.device           = device

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

    print("Evaluation...")
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

            # eval_loss = criterion(pred, ground_truth, gamma=config.gamma, alpha=config.alpha)

            # if not stopper_metric: # Just logging loss
            #     wandb.log({"running_val_loss": eval_loss})
            #     break
            # else: !
            # total_loss += float(eval_loss) * pred.numel() 
            # total_examples += pred.numel()
            # eval_loss = total_loss / total_examples

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
                # case "val_loss":
                #     eval_loss = total_loss / total_examples
                #     return eval_loss
                
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

            eval_loss = 1

    return mrr, eval_loss

def train(config, train_loader, val_loader, model, criterion, loss_name, optimizer, early_stopper, labels_type = 'usual', eval_period = 0, xp_name ="not indicated"):
    print("LOSS :", criterion)
    print("XP name :", xp_name)
    for epoch in range(config.epochs):
        total_loss = total_examples = 0

        for sampled_data in tqdm(train_loader, desc="Training"):

            sampled_data = sampled_data.to(device)
            pred = model(sampled_data, config)
            pos_samples  = torch.ones(len(sampled_data[config.labels['tail']].dst_pos_index), device=device)
            neg_samples  = torch.zeros(len(sampled_data[config.labels['tail']].dst_neg_index.view(-1)), device=device) # As many zeroes as there are negative samples * batch_size
            ground_truth = torch.cat((pos_samples, neg_samples))

            false_tails = sampled_data[config.labels['tail']].dst_neg_index.view(-1)
            true_tails = sampled_data['go']['dst_pos_index']
            true_tails = torch.repeat_interleave(true_tails,int(false_tails.size()[0]/true_tails.size()[0]))


            if labels_type == 'linsim':
                
                lin_neg_samples = pwc.lin_sims_for_batch(true_tails, false_tails)

                pos_samples = pos_samples.to(device)
                lin_neg_samples = lin_neg_samples.to(device)

                ground_truth = torch.cat((pos_samples, lin_neg_samples))

            if labels_type == 'rand_linsim':

                linsims = pwc.lin_sims_for_batch(true_tails, false_tails)
                lin_neg_samples = pwc.shuffle_tensor(linsims)
                lin_neg_samples = lin_neg_samples.to(device)
                pos_samples = pos_samples.to(device)

                ground_truth = torch.cat((pos_samples, lin_neg_samples))
                ground_truth.to(device)

            if labels_type == 'rand_ALL_labels_linsim':

                linsims = pwc.lin_sims_for_batch(true_tails, false_tails)
                linsims = linsims.to(device)
                pos_samples = pos_samples.to(device)

                ground_truth = torch.cat((pos_samples, linsims))
                ground_truth = pwc.shuffle_tensor(ground_truth)
                
                ground_truth.to(device)

            if labels_type == 'gaussian_noise':
                # Add gaussian noise to pos_samples to simulate noisy labels. Added noise must be negative and not exceed 1.
                ground_truth += torch.normal(mean=0, std=1, size=(len(ground_truth),), device=device).to(device)
                # apply sigmoid
                ground_truth = torch.sigmoid(ground_truth).to(device)

            if labels_type == 'usual':
                ground_truth = torch.sigmoid(ground_truth).to(device)

            if loss_name == "MSE":
                loss = criterion(pred, ground_truth)
            elif loss_name == "sigmoid focal":
                loss = criterion(pred, ground_truth, gamma=config.gamma, alpha=config.alpha)
            else :
                print("Unrecognized loss name !")

            wandb.log({"loss": loss})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        score, val_loss = evaluate(config, val_loader, model, criterion, stopper_metric=config.stopper_metric, compute_all_metrics= True)
        wandb.log({"val_loss": val_loss, f"{config.stopper_metric}": score})

        train_loss = total_loss / total_examples
        print(f"Epoch: {epoch:03d}, Avg. Loss: {train_loss:.10f}")


        
        
        # early_stopper(score)
        # if early_stopper.early_stop:
        #     print("Early stopping triggered at epoch", epoch)
        #     break

    print("Training done.")



for i in range(10):
      # Which labels should be used ?
      # Choices are : 'usual', 'rand_linsim', linsim', 'gaussian_noise', 'usual'.
      labels_type = 'rand_ALL_labels_linsim'
      loss_name = 'MSE'

      if loss_name == 'MSE':
        criterion = torch.nn.MSELoss()
      if loss_name == 'sigmoid focal':
        criterion = sigmoid_focal_loss
        

      # How many epochs between 2 time-consuming evaluations ?
      eval_period = 1

      # Wandb
      xp_name = 'GNN Baseline VS LinSim VS Noise With MSE'
      run_name = f'{labels_type} labels and {loss_name}'
      print(xp_name,':',run_name)

      run = wandb.init(project=xp_name, name = run_name)
      config = wandb.config
      config.labels_type = labels_type
      config.val_ratio=0.1
      config.homogeneous=False
      config.scorelist_size=1000
      config.split_ratio=0.8
      config.val_ratio=0.1
      config.test_ratio=0.1
      config.num_neighbors=[70,55,13,89,85]
      config.batch_size=1024
      config.train_neg_sampling_ratio=224
      config.epochs=epochs
      config.disjoint_train_ratio=0.6
      config.lr=0.0015308253347932983
      config.stopper_metric= 'hit_at_10'
      config.stopper_direction="maximize"
      config.stopper_patience=5
      config.stopper_frequency=1
      config.stopper_relative_delta=0.05
      config.gamma=1.3
      config.alpha=0.42680473078813763
      config.gnn_layer='ResGatedGraphConv'
      config.dropout=0.1
      config.norm='DiffGroupNorm'
      config.aggregation = 'min'
      config.hidden_channels=115
      config.num_layers=3
      config.attention_heads=4
      config.homogeneous = False
      config.labels = {'head' : 'genes', 'relation' : 'gene_ontology', 'tail' : 'go'}

      data = load_iric_data('/home/ebutz/ESL2024/data/full_iric/iric.csv', featureless=False)
      data  = T.ToUndirected(merge=True)(data) # Convert the graph to an undirected graph. Creates reverse edges for each edge.
      data = T.RemoveDuplicatedEdges()(data) # Remove duplicated edges
      print(data)
      print('data look valid : ',data.validate())

      train_data, val_data, test_data = split_data(data, config)
      train_loader, val_loader, test_loader = build_dataloaders(train_data, val_data, test_data, config)


      gnn_layers = get_gnn_layers(config)
      norm_layers = get_norm_layers(config, len(data.node_types))

      model = Model(config, data, norm_layers, gnn_layers).to(device)



      optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

      early_stopper = EarlyStopper(frequency=config.stopper_frequency, patience=config.stopper_patience,
                              direction=config.stopper_direction, relative_delta=config.stopper_relative_delta)

      train(config = config, train_loader = train_loader, val_loader =  val_loader, model = model,
            criterion = criterion, optimizer = optimizer, early_stopper = None, eval_period = 0, xp_name=run_name,
            loss_name=loss_name, labels_type=config.labels_type)


      # evaluate(config,  val_loader, model, criterion, compute_all_metrics=False, loader_type='validation', stopper_metric='hit_at_10')
      # evaluate(config, test_loader, model, criterion, compute_all_metrics=False, loader_type='test'      , stopper_metric='hit_at_10')


      wandb.finish()