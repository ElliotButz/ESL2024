#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from work_in_graph import *
from usefull_little_functions import *
import numpy as np
import torch


# In[2]:


epochs = 2

# How to calculate loss ?
# Choices are : 'usual', 'rand_linsim', linsim', 'gaussian_noise', 'usual'.
labels_type = 'usual'

# How many epochs between 2 time-consuming evaluations ?
eval_period = 1

# Wandb
xp_name = 'Trying family train'
run_name = f'family train for {labels_type} labels'
print(xp_name,':',run_name)

# Iric
iric_path       = '/home/ebutz/ESL2024/data/full_iric/iric.csv'

# # Ontology
# ontology_path = '/home/ebutz/ESL2024/data/go-basic.json.gz'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")


# In[3]:


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

# ------------- Load dataset ------------- #

data = load_iric_data(iric_path, featureless=False)
GO_edge_index = data['go', 'is_a', 'go'].edge_index # Important to extract ontology before making graph undirected
data  = T.ToUndirected(merge=True)(data)            # Convert the graph to an undirected graph. Creates reverse edges for each edge.
data = T.RemoveDuplicatedEdges()(data)              # Remove duplicated edges
# print(data)
print('data look valid : ',data.validate())

# ------------- Split dataset ------------- #

train_data, val_data, test_data = split_data(data, config)
train_loader, val_loader, test_loader = build_dataloaders(train_data, val_data, test_data, config)

# ------------- Loading ontology ------------- #

# nxo = from_file(ontology_path)
# nxo.freeze()
# pwc.nxo = nxo

df = pd.read_csv(iric_path, index_col=0, dtype=str)
df['source_node'] = df.index
df.columns = [x.lower() for x in df.columns]
go_map = get_nodelist(df, 'go')
go_to_idx = {node: i for i, node in enumerate(go_map)}
idx_to_go = {i: node for i, node in enumerate(go_map)}

# ------------- Making global variables accessibles to pwc ------------- #

pwc.map_to_GO        = idx_to_go
pwc.mapped_alt_tails = None
pwc.device           = device

# ------------- Model setup ------------- #

gnn_layers = get_gnn_layers(config)
norm_layers = get_norm_layers(config, len(data.node_types))

model = Model(config, data, norm_layers, gnn_layers).to(device)

# # ------------- Setup graph ------------- #

# G = nx.DiGraph()
# for s, d in zip(GO_edge_index[0].tolist(), GO_edge_index[1].tolist()):
#     G.add_edge(s, d)


# In[4]:


print('Nombre de Noeuds dans GO_edreg_index :', GO_edge_index.unique().size())


# In[5]:


train_data.generate_ids()
n_ids = train_data[config.labels['tail']].n_id
print('Nombre de Noeuds dans n_id :',n_ids.unique().size())
print(n_ids)
print(n_ids.size())
print(n_ids.unique().size())


# In[6]:



print('Nombre de Noeuds dans GO_edge_index :', GO_edge_index.unique().size())

# Génération des n_id :
train_data.generate_ids()
n_ids = train_data[config.labels['tail']].n_id
print('Nombre de Noeuds dans n_id :',n_ids.unique().size())


# In[7]:


all_n_ids = set(n_ids.unique().tolist())
all_in_GO_edge_index = set (GO_edge_index.unique().tolist())
print(len(all_n_ids - all_in_GO_edge_index))
print(len(all_in_GO_edge_index - all_n_ids))

list(all_in_GO_edge_index- all_n_ids)
list(all_n_ids - all_in_GO_edge_index)


# In[8]:


42969 in list(all_n_ids - all_in_GO_edge_index)


# In[ ]:





# In[10]:


# batchy['go']['dst_neg_index']


# In[11]:


# batchy


# In[12]:


train_data.generate_ids()
train_data[config.labels['tail']].n_id


# In[13]:


train_data[config.labels['tail']].n_id.size()


# In[14]:


GO_edge_index.size()


# In[15]:


src = in_onto = set(GO_edge_index[0].unique().tolist())
dst = in_onto = set(GO_edge_index[1].unique().tolist())
# print(dst-src)
# print(src-dst)


# In[16]:


def node_to_embed_id(node_id, node_to_embed_dict):
    try:
        return node_to_embed_dict[node_id]
    except :
        return 8533


# In[17]:


# GO_df = pd.DataFrame(GO_edge_index.numpy()[0], columns=['src_node_id'])
# GO_df['dist_node_id'] = GO_edge_index[1].numpy()
# GO_df['src_emb_id'] =GO_df.apply(lambda row : node_to_embed_id(row['src_node_id'], full_node_id_to_emb_id), axis=1)
# GO_df['dist_emb_id']=GO_df.apply(lambda row : node_to_embed_id(row['dist_node_id'], full_node_id_to_emb_id), axis=1)
# print(GO_df.head())
# EMB_GO_edge_index = torch.stack([torch.tensor(list(GO_df['src_emb_id'])), torch.tensor(list(GO_df['dist_emb_id']))])
# EMB_GO_edge_index


# In[18]:


def get_family(ontology_edge_index, max_dist):
    """
    Constructs a family graph up to a certain distance from a given ontology edge index.

    Parameters:
    - ontology_edge_index (torch.Tensor) : Edge index tensor representing the ontology graph.
    - max_dist (int)                     : Maximum distance to traverse to construct the family graph.      

    Returns:
    - df (pandas.DataFrame): DataFrame containing information about the family graph.
                                Columns:
                                    - source                   : The source node of the row.
                                    - {dist}-neighbors         : Set of neighbors at distance 'dist' from the source node.
                                    - {dist}-parents           : Set of parents at distance 'dist' from the source node.
                                    - {dist}-children          : Set of children at distance 'dist' from the source node.
                                    - family                   : Set of all neighbors up to distance 'max_dist' for the source node.
                                    - parents                  : Set of all parents up to distance 'max_dist' for the source node.
                                    - family_without_parents   : Set of family members excluding parents for the source node.
                                    - family_without_children  : Set of family members excluding children for the source node.
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
    df['1-neighbors'] = df.apply(lambda row : neighbors_for_nodes(graph = G, nodes = row['source']), axis=1)
    df['1-parents']   = df.apply(lambda row : predecessors_for_nodes(graph = G, nodes = row['source']), axis=1)
    df['1-children']  = df.apply(lambda row : successors_for_nodes(graph = G, nodes = row['source']), axis=1)

    for dist in range(2, max_dist+1):

        step = f'Distance {dist}/{max_dist}'
        print(step)

        tqdm.pandas(desc=f'Making dist {dist}-neighbors ')
        df[f'{dist}-neighbors'] = df.progress_apply(lambda row : neighbors_for_nodes(graph = G, nodes = row[f'{dist-1}-neighbors'])-set.union(*[row[f'{d}-neighbors'] for d in range(1,dist)])-row['source'],axis = 1)

        tqdm.pandas(desc=f'Making dist {dist}-parents   ')
        df[f'{dist}-parents'] = df.progress_apply(lambda row : predecessors_for_nodes(graph = G, nodes = row[f'{dist-1}-parents']),axis = 1)

        tqdm.pandas(desc=f'Making dist {dist}-children  ')
        df[f'{dist}-children'] = df.progress_apply(lambda row : successors_for_nodes(graph = G, nodes = row[f'{dist-1}-children']),axis = 1)
        
        # print(df)
    
    step = f'{max_dist}-family acquired.'
    print(step)
    tqdm.pandas(desc=f'Calculating family      ')
    df['family'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-neighbors'] for dist in range(1, max_dist+1)]) - (row['source']), axis = 1)
    
    tqdm.pandas(desc=f'Calculating children    ')
    df['children'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-children'] for dist in range(1, max_dist+1)]), axis = 1)

    tqdm.pandas(desc=f'Calculating parents     ')
    df['parents'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-parents'] for dist in range(1, max_dist+1)]), axis = 1)
    
    tqdm.pandas(desc=f'Calculating exclusion   ')
    df['family_without_parents'] = df.progress_apply(lambda row: row['family']- set.union(*[row[f'{dist}-parents'] for dist in range(1, max_dist+1)]), axis = 1)

    tqdm.pandas(desc=f'Calculating exclusion   ')
    df['family_without_children'] = df.progress_apply(lambda row: row['family']- set.union(*[row[f'{dist}-children'] for dist in range(1, max_dist+1)]), axis = 1)


    # tqdm.pandas(desc=f'Calculating lineage          ')
    # df['lineage'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-children'] for dist in range(1, max_dist+1)]+
    #                                                          [row[f'{dist}-parents'] for dist in range(1, max_dist+1)] +
    #                                                          [row['source']]),axis=1)
    # tqdm.pandas(desc=f'Calculating family - lineage ')
    # df['family_without_lineage'] = df.progress_apply(lambda row: set.union(*[row[f'{dist}-neighbors'] for dist in range(1, max_dist+1)]) - row['lineage'],axis=1)
    # print(df)

    # Re-ordering columns :
    dists = range(1, max_dist+1)
    df = df[['source'] + [f'{dist}-neighbors' for dist in dists]+
                         [f'{dist}-children' for dist in dists] +
                         [f'{dist}-parents' for dist in dists]  + 
                         ['family',
                          'parents', 
                          'family_without_parents', 
                          'family_without_children'
                          ]]
    
    return df

def plot_graph(edge_index):
    G = nx.DiGraph()
    G.add_edges_from(edge_index.t().tolist())

    pos = nx.spring_layout(G, iterations=50, seed = 1)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, arrows=True)
    plt.title("Directed Acyclic Graph (DAG)")
    plt.show()

edge_index = torch.tensor([[0, 1, 1, 2, 3, 3, 6, 15],   # Liste des départs des arêtes
                           [1, 2, 3, 3, 4, 5, 5, 6 ]],  # Liste des arrivées des arêtes
                          dtype=torch.long)
plot_graph(edge_index)
f = get_family(edge_index, 5)
print(f)


# In[19]:


GO_family_df = get_family(GO_edge_index, 3)


# In[49]:


GO_family_df


# In[52]:


l = list(GO_family_df.columns)
l.remove('source')
l


# In[59]:


def filter_set(s, threshold):
    return {x for x in s if x <= threshold}

cols = list(GO_family_df.columns)
cols.remove('source')

filterded_GO_family_df = GO_family_df[cols].map(lambda cell: filter_set(cell, 42964))
filterded_GO_family_df['source']=GO_family_df['source'].copy()


# In[60]:


filterded_GO_family_df.head()


# In[61]:


int(torch.Tensor([1]).item())


# In[62]:


def correct_numel(l, num_el, default_elements : list):

    '''
    Correct the number of elements in a list by adjusting its length.

    If the length of `l` matches `num_el`, the function returns `l` unchanged.
    If `l` is longer than `num_el`, it returns a sublist containing the first `num_el` elements.
    If `l` is shorter than `num_el`, it randomly selects elements from `l` to append to the end until the desired length is reached.
    If `l` is empty, it randomly selects elements from `default_elements` to create a list with `num_el` elements.

    Parameters:
        l (list)             : The input list.
        num_el (int)         : The desired number of elements in the list.
        default_elements (list): A list of elements to be used when `l` is empty.

    Returns:
        list: A list with the corrected number of elements.
    '''

    random.shuffle(l)

    if len(l)==0:
        print(f"No family negatives disponible - \nprobably looking for root's children or depraceted terms\n. Using random src terms as negatives.")
        l = random.choices(default_elements, k = num_el)
    elif len(l) == num_el:
        l = l
    elif len(l) > num_el:
        l = l[:num_el]  
    elif len(l) < num_el:
        l += random.choices(default_elements, k=num_el - len(l))
    
    return l


def make_family_negatives(family_df, neg_group_col: str, num_neg_per_pos):
    '''
    Generates negative examples for each positive example in a DataFrame.

    For each positive example in the DataFrame `family_df`, it creates negative examples by adjusting the number of elements in the specified column `neg_group_col`.

    Parameters:
        family_df (DataFrame): The input DataFrame containing positive examples.
        neg_group_col (str) : The column name representing groups of elements.
        num_neg_per_pos (int) : The desired number of negative examples per positive example.

    Returns:
        DataFrame: A DataFrame with negative examples generated for each positive example.
    '''
    if neg_group_col not in family_df.columns:
        raise KeyError(f'{neg_group_col} not in {list(family_df.columns)}')
    
    default_elements = list(family_df[neg_group_col].explode())
    
    tqdm.pandas(desc=f'Making family negatives')
    a = family_df.progress_apply(lambda row : torch.Tensor(correct_numel(list(row[neg_group_col]),num_el = num_neg_per_pos, default_elements=default_elements)), axis = 1)

    return a

def make_source_family_neg_dict(input_df, source_col = 'source', neg_col = 'family', num_neg_per_pos=224):
    '''
    Generates a dictionary with negative examples for each positive example based on a source column in an input DataFrame.

    Creates a dictionary where each key is a positive example from the source column, and each value is a list of negative examples generated from the corresponding positive example.

    Parameters:
        input_df (DataFrame): The input DataFrame containing positive and negative examples.
        source_col (str): The column name representing the source of positive examples.
        neg_col (str): The column name representing groups of elements.
        num_neg_per_pos (int): The desired number of negative examples per positive example.

    Returns:
        dict: A dictionary mapping positive examples to lists of negative examples.
    '''

    df = input_df[[source_col, neg_col]].copy()
    
    if not isinstance(df[source_col][0], np.int64):
        tqdm.pandas(desc=f'Correcting {source_col} type ')
        df[source_col]     = df.progress_apply(lambda row: int(torch.Tensor(list(row[source_col])).item()), axis = 1)

            
    if not isinstance(df[neg_col][0], torch.Tensor):
        tqdm.pandas(desc=f'Correcting {neg_col} type ')
        df[neg_col]     = df.progress_apply(lambda row: torch.Tensor(list(row[neg_col])), axis = 1)
    
    print('Making family negatives... Should last a few minutes')
    df['numeled_neg_col'] = make_family_negatives(df, neg_col, num_neg_per_pos)

    return df.set_index(source_col)['numeled_neg_col'].to_dict()

print(f)
example = make_source_family_neg_dict(input_df=f, source_col='source', neg_col='family_without_children', num_neg_per_pos=5)
filtered_f = f.map(lambda cell: filter_set(cell, 5))
print(filtered_f)
filtered_example = make_source_family_neg_dict(input_df=f, source_col='source', neg_col='family_without_children', num_neg_per_pos=5)
filtered_example


# In[63]:


source_family_neg_dict = make_source_family_neg_dict(input_df=filterded_GO_family_df, source_col='source', neg_col='family_without_children', num_neg_per_pos=224)


# In[65]:


# dist_families = {}
# for d in range(3,6):
#     dist_families[d]=get_family(GO_edge_index, max_dist = d)


# In[66]:


# df3 = get_family(GO_edge_index, max_dist = 3)
# df3.head()


# In[67]:


'''il faut modifier batchy['go']['dst_neg_index']'''
batchy = next(iter(train_loader))
batchy['go']['dst_neg_index']


# In[68]:


# batchy['go']['dst_neg_index'].size()


# In[69]:


# source_family_neg_dict = make_source_family_neg_dict(input_df=df3, source_col='source', neg_col='family_without_parents', num_neg_per_pos=batchy['go']['dst_neg_index'].size()[1])


# In[70]:


# print(batchy['go']['dst_neg_index'].size())
# print(batchy['go']['dst_pos_index'].size())


# In[71]:


source_family_neg_dict[0].size()


# In[72]:


for k in source_family_neg_dict.keys():
    source_family_neg_dict[k]


# In[73]:


def try_to_associate_family_neg(id, family_neg_dict, alt_values_list, num_neg):

    try :

        return family_neg_dict[id]
    except :
        print(f'{id} not found in dict, using random original terms for negatives.\n Probably a node that is not connected in GO because deprecated')
        return torch.tensor(random.choices(alt_values_list, k = num_neg))
    

def make_family_neg_for_batch(batch, family_neg_dict):
    
    num_neg = batch['go']['dst_neg_index'].size()[1]
    alt_values = list(set(batch['go']['dst_neg_index'].unique().tolist()) - set(list(range(42960,42999))))

    pos = batch['go']['dst_pos_index'].numpy()
    df = pd.DataFrame(pos, columns=['pos'])

    df['family_neg'] = df.apply(lambda row : try_to_associate_family_neg(id              =  row['pos'],
                                                                         family_neg_dict = family_neg_dict,
                                                                         num_neg         = num_neg,
                                                                         alt_values_list = alt_values),
                                            axis =1)
    

    return torch.stack(df['family_neg'].to_list()).type(dtype=torch.LongTensor)

        
    

batchy_family_neg = make_family_neg_for_batch(batchy, source_family_neg_dict)
print('Generated negs :',batchy_family_neg.size(), batchy_family_neg.dtype)
print('Original negs  :',batchy['go']['dst_neg_index'].size(), batchy_family_neg.dtype)


# In[76]:


for k in tqdm(source_family_neg_dict.keys()):
    for i in source_family_neg_dict[k]:
        if i<0 :
            print(i)
        elif i >42965:
            print(i)


# In[ ]:


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

    return mrr, eval_loss

@timer_func
def train(config, train_loader, val_loader, model, criterion, optimizer, early_stopper, labels_type = 'usual', eval_period = 0):
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

                lin_ground_truth = torch.cat((pos_samples, lin_neg_samples))
                loss = criterion(pred, lin_ground_truth, gamma=config.gamma, alpha=config.alpha)

            if labels_type == 'rand_linsim':

                linsims = pwc.lin_sims_for_batch(true_tails, false_tails)
                lin_neg_samples = pwc.shuffle_tensor(linsims)
                lin_neg_samples.to(device)
                
                lin_ground_truth = torch.cat((pos_samples, lin_neg_samples))
                loss = criterion(pred, lin_ground_truth, gamma=config.gamma, alpha=config.alpha)

            if labels_type == 'gaussian_noise':
                # Add gaussian noise to pos_samples to simulate noisy labels. Added noise must be negative and not exceed 1.
                ground_truth += torch.normal(mean=0, std=1, size=(len(ground_truth),), device=device)
                # apply sigmoid
                ground_truth = torch.sigmoid(ground_truth)
                loss = criterion(pred, ground_truth, gamma=config.gamma, alpha=config.alpha)

            if labels_type == 'usual':
                ground_truth = torch.sigmoid(ground_truth)
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
        if eval_period:
            if epoch%eval_period == 0:
                # Compute metrics and check for early stopping
                score, val_loss = evaluate(config, val_loader, model, criterion, stopper_metric=config.stopper_metric, compute_all_metrics= True)
                wandb.log({"avg_loss": train_loss, "val_loss": val_loss, f"{config.stopper_metric}": score})

        # early_stopper(score)
        # if early_stopper.early_stop:
        #     print("Early stopping triggered at epoch", epoch)
        #     break

    print("Training done.")

    return model

def train_with_family(family_neg_dict, config, train_loader, val_loader, model, criterion, optimizer, early_stopper, labels_type = 'usual', eval_period = 0):
    
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        i  = 0
        for sampled_data in tqdm(train_loader, desc="Training"):

            sampled_data_family_neg = make_family_neg_for_batch(sampled_data, family_neg_dict)
            print(sampled_data_family_neg)
            sampled_data['go']['dst_neg_index'] = sampled_data_family_neg
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

                lin_ground_truth = torch.cat((pos_samples, lin_neg_samples))
                loss = criterion(pred, lin_ground_truth, gamma=config.gamma, alpha=config.alpha)

            if labels_type == 'rand_linsim':

                linsims = pwc.lin_sims_for_batch(true_tails, false_tails)
                lin_neg_samples = pwc.shuffle_tensor(linsims)
                lin_neg_samples.to(device)
                
                lin_ground_truth = torch.cat((pos_samples, lin_neg_samples))
                loss = criterion(pred, lin_ground_truth, gamma=config.gamma, alpha=config.alpha)

            if labels_type == 'gaussian_noise':
                # Add gaussian noise to pos_samples to simulate noisy labels. Added noise must be negative and not exceed 1.
                ground_truth += torch.normal(mean=0, std=1, size=(len(ground_truth),), device=device)
                # apply sigmoid
                ground_truth = torch.sigmoid(ground_truth)
                loss = criterion(pred, ground_truth, gamma=config.gamma, alpha=config.alpha)

            if labels_type == 'usual':
                ground_truth = torch.sigmoid(ground_truth)
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
        if eval_period:
            if epoch%eval_period == 0:
                # Compute metrics and check for early stopping
                score, val_loss = evaluate(config, val_loader, model, criterion, stopper_metric=config.stopper_metric, compute_all_metrics= True)
                wandb.log({"avg_loss": train_loss, "val_loss": val_loss, f"{config.stopper_metric}": score})

        # early_stopper(score)
        # if early_stopper.early_stop:
        #     print("Early stopping triggered at epoch", epoch)
        #     break

    print("Training done.")

model = Model(config, data, norm_layers, gnn_layers).to(device)
criterion = sigmoid_focal_loss
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
early_stopper = EarlyStopper(frequency=config.stopper_frequency, patience=config.stopper_patience,
                             direction=config.stopper_direction, relative_delta=config.stopper_relative_delta)

# trained_model = train(config, train_loader, val_loader, model, criterion, optimizer, early_stopper, labels_type=config.labels_type, eval_period = eval_period)

retrained_model = train_with_family(source_family_neg_dict , config, train_loader, val_loader, model, criterion, optimizer, early_stopper, labels_type=config.labels_type, eval_period = eval_period)


# In[ ]:


# trained_model = train(config, train_loader, val_loader, model, criterion, optimizer, early_stopper, labels_type=config.labels_type, eval_period = eval_period)

# evaluate(config,  val_loader, model, criterion, compute_all_metrics=False, loader_type='validation', stopper_metric='hit_at_10')
# evaluate(config, test_loader, model, criterion, compute_all_metrics=False, loader_type='test'      , stopper_metric='hit_at_10')

# retrained_model = train_with_family(source_family_neg_dict , config, train_loader, val_loader, model, criterion, optimizer, early_stopper, labels_type=config.labels_type, eval_period = eval_period)

# evaluate(config,  val_loader, model, criterion, compute_all_metrics=False, loader_type='validation', stopper_metric='hit_at_10')
# evaluate(config, test_loader, model, criterion, compute_all_metrics=False, loader_type='test'      , stopper_metric='hit_at_10')

wandb.finish

