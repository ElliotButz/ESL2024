import pandas as pd
import torch
import torch_geometric
import torch.optim as optim
from torch_geometric.nn import ComplEx
import torch.nn.functional as F
import nxontology as nxo
import copy
import wandb
import tqdm

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
                           xp_name:str = '',
                           run_name: str = '',
                           use_wandb = False,
                           epochs: int = 1000,
                           batch_size: int = 4096,
                           eval_period = 2,
                           reset_parameters = False,
                           params_save_path = '',
                           device = 'cuda',
                           dataset_name = 'iric'
                           ):
    
    hidden_channels = model.hidden_channels
    print(f"Train and test.\nModel type : {type(model)}\nHidden channels : 6 * {hidden_channels}\nEpochs : {epochs}\nEval period : {eval_period}\nUse WanDB : {use_wandb}")
    print(f"Device : {device}", "Could reach GPU :", torch.Tensor([0,1]).to(device).is_cuda)
    print(f"Evaluation each {eval_period} epochs.")
    
    
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
            name=run_name,
            config={
            "architecture": str(type(model)),
            "dataset": dataset_name,
            "epochs": epochs,
            'hidden_channels' : hidden_channels,
            'batch_size' : batch_size
            }
        )

    # ----------------------------------------------- Train and eval

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
        print(f'Loss on train set : {loss:.4f}\nLoss on test set  : {test_loss:.4f}')

    # ----------------------------------------------- Periodic Evaluation
        if eval_period:
            if epoch%eval_period == 0:
                print('Eval...')
                rank, mrr, hits = test(test_data, model=model, device=device, batch_size=batch_size)

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


def lin_sim_on_mapped_terms(mapped_term1, mapped_term2):
    term1 = map_to_GO[mapped_term1]
    term2 = map_to_GO[mapped_term2]
    if (term1 in nxo.graph._node and term2 in nxo.graph._node):
        sim = nxo.similarity(term1, term2).lin 
        return sim
    else:
        return 0  

def best_lim_sim_for_triple(head, rel, tail)-> torch.Tensor:
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

    head_index,rel_type,tail_index = head_index.cpu(),rel_type.cpu(),tail_index.cpu()
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

    Moreover, test() is made on 1000 triples instead of all existing triples.
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
    
    def test(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = True,
    ) -> tuple[float, float, float]:
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
        """
        # Instead of :
        # arange = range(head_index.numel())
        # I set : 
        arange = range(1000)
        # arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            rank = int((torch.cat(scores).argsort(
                descending=True) == t).nonzero().view(-1))
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k
    
# class ComplEx_with_LinSim_labels(tail_only_ComplEx):
#   def loss(
#             self,
#             head_index: torch.Tensor,
#             rel_type: torch.Tensor,
#             tail_index: torch.Tensor,
#             ) -> torch.Tensor:
            
#         '''
#         tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
#         '''

#         pos = head_index, rel_type, tail_index
#         neg = self.random_sample(head_index, rel_type, tail_index)


#         pos_score = self(*pos)
#         neg_score = best_lin_sims_for_batch(*neg)
#         neg_score.to(device)
#         scores = torch.cat([pos_score, neg_score], dim=0)

#         pos_target = torch.ones_like(pos_score).to(device)
#         neg_target = torch.zeros_like(neg_score).to(device)
#         target = torch.cat([pos_target, neg_target], dim=0)

#         return F.binary_cross_entropy_with_logits(scores, target)

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
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = best_lin_sims_for_batch(*neg).to(device)

        target = torch.cat([pos_target, neg_target], dim=0)
        print(target)

        return F.binary_cross_entropy_with_logits(scores, target)

class ComplEx_with_LinSim_labels_and_usual_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        '''

        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        pos_target = torch.ones_like(pos_score).to(device)
        neg_target = torch.zeros_like(neg_score).to('cpu')
        lin_neg_target = best_lin_sims_for_batch(*neg).to(device)
        neg_target = neg_target.to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, lin_target) 

class ComplEx_with_RANDOMISED_LinSim_labels_and_usual_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        not-1 labels are shuffled.
        '''

        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        pos_target = torch.ones_like(pos_score).to(device)
        neg_target = torch.zeros_like(neg_score).to('cpu')
        lin_neg_target = shuffle_tensor(best_lin_sims_for_batch(*neg)).to(device)
        neg_target = neg_target.to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, lin_target) 

class ComplEx_with_normal_noise_and_usual_labels(ComplEx_with_LinSim_labels_and_usual_labels):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        ComplEx_with_LinSim_labels_and_usual_loss.loss() modified to have a normal(0,1) noise instead of Lin labels.
        '''

        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)

        neg = self.random_sample(head_index, rel_type, tail_index)
        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        pos_target = torch.ones_like(pos_score).to(device)
        neg_target = torch.zeros_like(neg_score).to('cpu')
        # I replace this line :
        # lin_neg_target = best_lin_sims_for_batch(*neg).to(device)
        # With :
        lin_neg_target = torch.rand(size = neg_target.size()).to(device)
        neg_target = neg_target.to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, lin_target) 

class best_LinSim_ComplEx(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to account a LinSim term :
        one withdraw the mean(bests similarities between each false tail of a triple to its possible tails) to the loss.
        '''

        pos = head_index, rel_type, tail_index

        false_head_index, false_rel_type, false_tail_index = self.random_sample(head_index, rel_type, tail_index)
        neg = false_head_index, false_rel_type, false_tail_index

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score) 
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        # Calculating LinSim(positive_head, negative_head) : 
        similarities = best_lin_sims_for_batch(head_index, rel_type, false_tail_index)

        return F.binary_cross_entropy_with_logits(scores, target) - torch.mean(similarities)

class LinSim_ComplEx(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to account a LinSim term : one simply withdraw mean(similarities(batch)) to the loss.
        '''

        pos = head_index, rel_type, tail_index

        false_head_index, false_rel_type, false_tail_index = self.random_sample(head_index, rel_type, tail_index)
        neg = false_head_index, false_rel_type, false_tail_index

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score) 
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        # stacking true and falses tails in df :
        pos_and_neg_tails = pd.DataFrame(torch.stack((tail_index,false_tail_index)).transpose(0,1)).astype("int")

        # Calculating LinSim(positive_head, negative_head) : 
        similarities = torch.tensor(pos_and_neg_tails.apply(lambda row : lin_sim_on_mapped_terms(row[0], row[1]),
                                                      axis = 1).values
                                    )

        return F.binary_cross_entropy_with_logits(scores, target) - torch.mean(similarities)

class LinSim_Only_ComplEx(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to account a LinSim term : one simply withdraw mean(similarities(batch)) to the loss.
        '''

        pos = head_index, rel_type, tail_index

        false_head_index, false_rel_type, false_tail_index = self.random_sample(head_index, rel_type, tail_index)
        neg = false_head_index, false_rel_type, false_tail_index

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score) 
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        # stacking true and falses tails in df :
        pos_and_neg_tails = pd.DataFrame(torch.stack((tail_index,false_tail_index)).transpose(0,1)).astype("int")

        # Calculating LinSim(positive_head, negative_head) : 
        similarities = torch.tensor(pos_and_neg_tails.apply(lambda row : lin_sim_on_mapped_terms(row[0], row[1]),
                                                      axis = 1).values
                                    )

        return torch.tensor([1])- torch.mean(similarities)