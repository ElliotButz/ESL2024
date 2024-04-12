print("\nImports...")

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import pickle
import os
import sys
sys.path.append( '/home/ebutz/ESL2024/code/utils' )
import play_with_complex as pwc


import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import ComplEx
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


from nxontology.imports import from_file

# ------------- Paths ------------- #

# Iric
mapped_iric_path  = '/home/ebutz/ESL2024/data/full_iric/altailed_mapped_iric.pickle'
altails_dict_path = '/home/ebutz/ESL2024/data/full_iric/altail_iric_DICT.pickle'

# Model to train :
hidden_channels = 15
batch_size      = 4096
epochs          = 70
eval_period     = 2
lin_factor      = 1

params_save_name = f"PARAMS_ComplEx_6_times_{hidden_channels}_HC_{epochs}_epochs_{batch_size}_BS_on_full_iric"
model_parameters_path = "/home/ebutz/ESL2024/data/mapping_datasets_and_model_for_genes_to_phenotypes_iric/"+params_save_name

# Ontology
ontology_path = "/home/ebutz/ESL2024/data/go-basic.json.gz"

check_dicts = False

# ------------- Cuda ------------- #

print("\nCuda check...")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
print("Could reach GPU :", torch.Tensor([0,1]).to(device).is_cuda)


# ------------- Loading datas ------------- #

print("\nLoading iric...")
mapped_iric = pd.read_pickle(mapped_iric_path)
print(mapped_iric.head())
print('mapped_alt_tails type :', type(mapped_iric.iloc[0]['mapped_alt_tails']))

GO_to_map = mapped_iric.set_index('object')['mapped_object'].to_dict()
map_to_GO = {value: key for key, value in GO_to_map.items()}

if check_dicts:
    looks_ok: bool = True
    for i in tqdm(range(len(list(mapped_iric['object']))), desc = "Checking GO to MAP dict"):
        if GO_to_map[mapped_iric['object'][i]]!=mapped_iric['mapped_object'][i] :
            looks_ok = False
    print('GO - Mapping dicts looks ok :', looks_ok)

with open(altails_dict_path, 'rb') as handle:
    mapped_alt_tails = pickle.load(handle)
print("Alternative tails dict (first key-value pair):", list(mapped_alt_tails.items())[0])

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
                            is_undirected = False,
                            add_negative_train_samples = False,
                            )

train_set, val_set, test_set = transform(iric_pyg)
print('Dataset splits look valid (train, val, test):',train_set.validate(raise_on_error = True),
                                                      val_set.validate(raise_on_error   = True),
                                                      test_set.validate(raise_on_error  = True))

# ------------- Loading ontology ------------- #

print("\nLoading ontology...")
nxo = from_file(ontology_path)
nxo.freeze()

# ------------- Making global variables accessibles to pwc ------------- #

pwc.map_to_GO        = map_to_GO
pwc.nxo              = nxo
pwc.mapped_alt_tails = mapped_alt_tails
pwc.device           = device


# ------------- Init model ------------- #

tail_only = pwc.tail_only_ComplEx(
                                num_nodes       = train_set.num_nodes,
                                num_relations   = train_set.edge_index.size()[1],
                                hidden_channels = hidden_channels,
                                ).to(device)

LinSim_labels = pwc.ComplEx_with_LinSim_labels(
                                num_nodes       = train_set.num_nodes,
                                num_relations   = train_set.edge_index.size()[1],
                                hidden_channels = hidden_channels,
                                ).to(device)

usual_and_LinSim_labels = pwc.ComplEx_with_LinSim_labels_and_usual_labels(
                                num_nodes       = train_set.num_nodes,
                                num_relations   = train_set.edge_index.size()[1],
                                hidden_channels = hidden_channels,
                                ).to(device)

usual_labels_and_RANDOMISED_LinSim_labels = pwc.ComplEx_with_RANDOMISED_LinSim_labels_and_usual_labels(
                                num_nodes       = train_set.num_nodes,
                                num_relations   = train_set.edge_index.size()[1],
                                hidden_channels = hidden_channels,
                                ).to(device)

usual_labels_and_gaussian_noise_labels = pwc.ComplEx_with_normal_noise_and_usual_labels(
                                num_nodes       = train_set.num_nodes,
                                num_relations   = train_set.edge_index.size()[1],
                                hidden_channels = hidden_channels,
                                ).to(device)




# # ------------- Train and evaluate ------------- #

# pwc.train_and_test_complex(model      = tail_only,
#                            train_data = train_set,
#                            test_data  = test_set,
#                            device     = device,
#                            use_wandb  = True,
#                            xp_name    = 'Add LinSim labels vs Noise Labels on full dataset',
#                            run_name   = 'Baseline (Usual ComplEx tail only)',
#                            eval_period= eval_period
#                            )

pwc.train_and_test_complex(model      = LinSim_labels,
                           train_data = train_set,
                           test_data  = test_set,
                           device     = device,
                           use_wandb  = True,
                           xp_name    = 'Add LinSim labels vs Noise Labels on full dataset',
                           run_name   = 'LinSim_labels',
                           eval_period= eval_period
                           )

# pwc.train_and_test_complex(model      = summed_labels_with_LinSim,
#                            train_data = train_set,
#                            test_data  = test_set,
#                            device     = device,
#                            use_wandb  = True,
#                            xp_name    = 'Add LinSim labels vs Noise Labels on full dataset',
#                            eval_period= eval_period
#                            )

# pwc.train_and_test_complex(model      = summed_labels_with_noise,
#                            train_data = train_set,
#                            test_data  = test_set,
#                            device     = device,
#                            use_wandb  = True,
#                            xp_name    = 'Add LinSim labels vs Noise Labels on full dataset',
#                            eval_period= eval_period
#                            )

