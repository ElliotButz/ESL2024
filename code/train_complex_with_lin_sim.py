print("\nImports...")

import sys
sys.path.append("/home/ebutz/ESL2024/code/utils")

import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas()

import ast
import random

import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import ComplEx
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from nxontology.imports import from_file
import os

import play_with_complex as pwc

import wandb


# ------------- Paths ------------- #

# Iric
mapped_iric_path = '/home/ebutz/ESL2024/data/mapped_iric.tsv'

# Model to train :
hidden_channels = 15
batch_size = 4096
epochs = 1000
lin_factor = 1

params_save_name = f"PARAMS_ComplEx_6_times_{hidden_channels}_HC_{epochs}_epochs_{batch_size}_BS_on_full_iric"
model_parameters_path = "/home/ebutz/ESL2024/data/mapping_datasets_and_model_for_genes_to_phenotypes_iric/"+params_save_name

# Ontology
ontology_path = "/home/ebutz/ESL2024/data/go-basic.json.gz"

# ------------- Cuda ------------- #

print("\nCuda check...")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
print("Could reach GPU :", torch.Tensor([0,1]).to(device).is_cuda)


# ------------- Loading datas ------------- #

print("\nLoading iric...")

mapped_iric = pd.read_csv(mapped_iric_path,
                          sep = '\t'
                          )



# row1 = mapped_iric.iloc[0]
# print(type(row1['mapped_alt_tails']))
print(mapped_iric.head())
# GO_to_map = mapped_iric.set_index('object')['mapped_object'].to_dict()
# map_to_GO = {value: key for key, value in GO_to_map.items()}

# # Checking dict :
# looks_ok: bool = True
# print('Number of triples in graph :', len(list(mapped_iric['object'])))
# for i in range(len(list(mapped_iric['object']))):
#     if GO_to_map[mapped_iric['object'][i]]!=mapped_iric['mapped_object'][i] :
#         looks_ok = False
# print('GO - Mapping dicts looks ok :', looks_ok)

# # Making dict of alternatives tails :
# mapped_alt_tails = {}
# for index, row in mapped_iric.iterrows():
#     key = (row['mapped_subject'], row['mapped_predicate'])
#     if key not in mapped_alt_tails:
#         mapped_alt_tails[key] = set()
#         print(set(row['mapped_alt_tails'].values()))

#         try :
#             mapped_alt_tails[key].update(set(row['mapped_alt_tails']))
#         except:
#             # print("Problem with :",row['mapped_alt_tails'])
#             pass

# for key, value in mapped_alt_tails.items():
#     # mapped_alt_tails[key]=np.array(list(value))
#     mapped_alt_tails[key]=list(value)


# print(list(mapped_alt_tails.items())[0])

# # ------------- Making datasets ------------- #

print("\nMaking datasets...")
# Edges index :
heads = list(mapped_iric['mapped_subject'])
tails = list(mapped_iric['mapped_object'])
edge_index = torch.tensor([heads,tails], dtype=torch.long)
# edges attributes :
edge_attributes = torch.tensor(mapped_iric['mapped_predicate'])


iric_pyg = Data(
                num_nodes = len(set(mapped_iric['object']).union(set(mapped_iric['subject']))),
                edge_index = edge_index,
                edge_attr = edge_attributes
                )

print(iric_pyg)

print("\nDataset looks valid :",iric_pyg.validate(raise_on_error=True))

transform = RandomLinkSplit(
                            num_val = 0.1,
                            num_test = 0.1,
                            is_undirected=False,
                            add_negative_train_samples=False,
                            )

train_set, val_set, test_set = transform(iric_pyg)
print('Dataset splits look valid (train, val, test):',train_set.validate(raise_on_error=True),
                                                      val_set.validate(raise_on_error=True),
                                                      test_set.validate(raise_on_error=True))

# # ------------- Loading ontology ------------- #

print("\nLoading ontology...")
nxo = from_file(ontology_path)
nxo.freeze()

# # ------------- Init model ------------- #
complex = ComplEx(num_nodes=train_set.num_nodes,
                  num_relations = train_set.edge_index.size()[1],
                  hidden_channels=hidden_channels,
                  ).to(device)

# # ------------- Train and evaluate ------------- #

pwc.train_and_test_complex(model = complex,
                           train_data = train_set,
                           test_data = test_set,
                           use_wandb=False)