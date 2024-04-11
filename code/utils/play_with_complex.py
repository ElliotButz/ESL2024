import pandas as pd
import torch
import torch_geometric
import torch.optim as optim
from torch_geometric.nn import ComplEx
import nxontology as nxo
import copy
import wandb

def train(loader, model, optimizer, device):
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:

        head_index, rel_type, tail_index = head_index.to(device), rel_type.to(device), tail_index.to(device)
        
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


def train_with_similarity_shrinkage(loader, model, optimizer, lin_term, ontology, device):

    '''
    Shrinks the weights modification operated by SGD.
    Shrinkage is lesser if Lin term is high (close to 1)

    lin_term should have its values between 0 and 1.
    Requires model.loss() to return the false tail_index generated.
    Requires lin_term's args to be head_index, rel_type, tail_index, false_tail_index, ontology.

    Should not work : Explore the same landscape.
    '''
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        head_index, rel_type, tail_index = head_index.to(device), rel_type.to(device), tail_index.to(device)
        optimizer.zero_grad()

        before = copy.deepcopy(model.state_dict())
        loss, false_tail_index = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        after = model.state_dict()

        for key, params in after.items():
            after[key] += lin_term(head_index, rel_type, tail_index, false_tail_index, ontology) * (after[key]-before[key])
        model.load_state_dict(after)

        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()

    return total_loss / total_examples

@torch.no_grad()
def test(data, model, 
         batch_size,
         device,
         k = 10):
    model.eval()
    return model.test(
        head_index=data.edge_index[0].to(device),
        tail_index=data.edge_index[1].to(device),
        rel_type=data.edge_attr.to(device),
        batch_size=batch_size,
        k=k, #The k in Hit@k
    )

def get_test_loss(loader, model, device):

    model.eval()
    total_loss = total_examples = 0

    for head_index, rel_type, tail_index in loader:

        head_index, rel_type, tail_index = head_index.to(device), rel_type.to(device), tail_index.to(device)
        loss = model.loss(head_index, rel_type, tail_index)
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()

    return total_loss / total_examples

def train_and_test_complex(
                           model,
                           train_data: torch_geometric.data.data.Data,
                           test_data : torch_geometric.data.data.Data,
                           hidden_channels:int = None,
                           xp_name:str = '',
                           epochs: int = 1000,
                           batch_size: int = 4096,
                           eval_period = 500,
                           reset_parameters = False,
                           params_save_path = '',
                           use_wandb = False,
                           device = 'cpu',
                           dataset_name = 'iric'
                           ):
    
    
    # ----------------------------------------------- Reset parameters
    if reset_parameters :
        model.reset_parameters()

    # ----------------------------------------------- Loader
    loader = model.loader(
                          head_index = train_data.edge_index[0],
                          tail_index = train_data.edge_index[1],
                          rel_type   = train_data.edge_attr,
                          batch_size = batch_size,
                          shuffle    = True)
    test_loader = model.loader(
                          head_index = test_data.edge_index[0],
                          tail_index = test_data.edge_index[1],
                          rel_type   = test_data.edge_attr,
                          batch_size = batch_size,
                          shuffle    = True)

    # ----------------------------------------------- Optimizer
    optimizer = optim.Adam(model.parameters())

    # ----------------------------------------------- WandB
    if use_wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=xp_name,
            
            config={
            "architecture": str(type(model)),
            "dataset": dataset_name,
            "epochs": epochs,
            'hidden_channels' : hidden_channels,
            'batch_size' : batch_size
            }
        )

    # ----------------------------------------------- Train and eval
    print('Train...')

    torch.set_grad_enabled(True)
    model.to(device)

    train_losses = []
    test_losses = []
    for epoch in range(0, epochs+1):

        loss = train(loader = loader,
                     model  = model,
                     optimizer = optimizer,
                     device = device)
        test_loss = get_test_loss(
                            loader = test_loader,
                            model  = model,
                            device = device)
        
        train_losses.append(loss)
        test_losses.append(test_loss)
        
        if use_wandb : 
            wandb.log({"loss": loss, "loss on test": test_loss})
        print(f'Epoch: {epoch:03d}')
        print('Loss on train set : {loss:.4f}\nLoss on test set  : {test_loss:.4f}')

    # ----------------------------------------------- Periodic Evaluation
        if eval_period:
            if epoch%eval_period == 0:
                print('Eval...')
                rank, mrr, hits = test(test_data, model=model, device=device)

                print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}',
                    f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')
                if use_wandb:
                    wandb.log({"Val Mean Rank" : rank,
                            "Val MRR" : mrr,
                            "hits@10": hits})

    # ----------------------------------------------- End WandB
    if use_wandb:
        print('End wandb...')
        wandb.finish()
        print("WandB finished.")

    # ----------------------------------------------- Model to cpu
    model.to('cpu')
    
    # ----------------------------------------------- Save model
    if params_save_path != '':
        torch.save(model.state_dict(), params_save_path)
        print("Model saved at", params_save_path)

    print("End")

    return model, train_losses, train_losses


def lin_sim_on_mapped_terms(mapped_term1, mapped_term2, map_to_GO:dict):
    term1 = map_to_GO[mapped_term1]
    term2 = map_to_GO[mapped_term2]
    if (term1 in nxo.graph._node and term2 in nxo.graph._node):
        sim = nxo.similarity(term1, term2).lin 
        return sim
    else:
        return 0  

def best_lim_sim_for_triple(head, rel, tail, mapped_alt_tails:dict, map_to_GO)-> torch.Tensor:
    max_lin_sim=0
    for alt_tail in mapped_alt_tails[(head, rel)]:

        if (map_to_GO[tail] in nxo.graph._node
            and
            map_to_GO[alt_tail] in nxo.graph._node):
                
                sim = nxo.similarity(map_to_GO[tail], map_to_GO[alt_tail]).lin # Pourrait être amélioré : actuellement, on calcule plein de similarités différentes.
                
                if max_lin_sim < sim < 1:
                    max_lin_sim = sim
    
    return max_lin_sim

def best_lin_sims_for_batch(head_index:torch.Tensor, rel_type:torch.Tensor, tail_index:torch.Tensor):

    batch = pd.DataFrame(torch.transpose(torch.stack((head_index,rel_type,tail_index)),
                                         0,1)
                        )
    
    return torch.Tensor(batch.apply(lambda row : best_lim_sim_for_triple(head=row[0],
                                                                         rel=row[1],
                                                                         tail=row[2]),
                                    axis=1))

def lin_sims_for_batch(term1: torch.Tensor, term2: torch.Tensor)->torch.Tensor:

    batch = pd.DataFrame(torch.transpose(torch.stack((term1, term2)),
                                         0,1)
                        )    
    return torch.Tensor(batch.apply(lambda row : best_lim_sim_for_triple(head=row[0],
                                                                         rel=row[1],
                                                                         tail=row[2]),
                                    axis=1))

def shuffle_tensor(t: torch.Tensor):
    '''
    Shuffles elements of a tensor.
    WARNING :
    shuffle_tensor(torch.tensor([[0,1,2,3,4,5],[6,7,8,9,0,1]])) returns :
    tensor([[0, 1, 2, 3, 4, 5],  OR tensor([[6, 7, 8, 9, 0, 1],
            [6, 7, 8, 9, 0, 1]])            [0, 1, 2, 3, 4, 5]])
    '''
    idx = torch.randperm(t.shape[0])
    return t[idx].view(t.size())

class tail_only_ComplEx(ComplEx):

    '''
    Overwritting random_sample() to make negative triples by setting a random tail to each triple,
    instead of setting a random head or tail.
    '''
    @torch.no_grad()
    def random_sample(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        ):

        """
        Randomly samples negative triplets by replacing the tail.
        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """

        tail_index = shuffle_tensor(tail_index.clone())

        return head_index, rel_type, tail_index
    
class ComplEx_with_LinSim_labels(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = best_lin_sims_for_batch(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score) 
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)

class ComplEx_with_LinSim_labels_and_usual_loss(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        '''

        pos = head_index, rel_type, tail_index

        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score) 
        neg_target = torch.zeros_like(neg_score)
        lin_neg_target = best_lin_sims_for_batch(*neg)

        target = torch.cat([pos_target, neg_target], dim=0)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, lin_target) 