import torch
import os
import pickle
import sys
sys.path.append( '/home/ebutz/ESL2024/code/utils' )
import play_with_complex as pwc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

a = torch.Tensor([0,1])
print(a.is_cuda)

a=  a.to(device)
print(a.is_cuda)

print(pwc.shuffle_tensor(torch.tensor([0,1,2,3,4,5,6,7,8,9,10])))
