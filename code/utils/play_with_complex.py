from tqdm import tqdm
import random 
import pandas as pd
import torch
import torch_geometric
import torch.optim as optim
from torch_geometric.nn import ComplEx
import torch.nn.functional as F
import nxontology as nxo
import copy
import wandb


def train_model(model_name:str, hidden_channels_list: list,  epochs:int, eval_period:int, 
                device:str,
                use_wandb:bool, xp_name :str,
                test_set:torch_geometric.data.data.Data,train_set: torch_geometric.data.data.Data,
                file_import : str):
    
    for hidden_channels in hidden_channels_list :
        xp_name = f"Lin VS Baseline with {hidden_channels}*6 HC on {epochs} epochs"

        model_class = getattr(file_import, model_name)

        model = model_class(
            num_nodes       = train_set.num_nodes,
            num_relations   = train_set.edge_index.size()[1],
            hidden_channels = hidden_channels,
        ).to(device)

        train_and_test_complex(
            model       = model,
            train_data  = train_set,
            test_data   = test_set,
            device      = device,
            use_wandb   = use_wandb,
            xp_name     = xp_name,
            run_name    = model_name,
            eval_period = eval_period,
            epochs      = epochs
        )

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

def get_test_loss(loader, model, device):

    model.eval()
    total_loss = total_examples = 0

    for head_index, rel_type, tail_index in loader:

        head_index, rel_type, tail_index = head_index.to(device), rel_type.to(device), tail_index.to(device)
        loss = model.loss(head_index, rel_type, tail_index)
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()

    return total_loss / total_examples

def train_and_test_complex(model,
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
                           dataset_name = 'iric',
                           k = 10
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
    print(f'Model on {device}.')
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
        
        # if use_wandb : 
            # wandb.log({"loss": loss, "loss on test": test_loss})
        print(f'Epoch: {epoch:03d}/{epochs}')
        print(f'Loss on train set : {loss:.4f}\nLoss on test set  : {test_loss:.4f}')

    # ----------------------------------------------- Periodic Evaluation
        if eval_period:
            if epoch%eval_period == 0:
                print('Eval...')
                model.eval()
                batch = next(iter(test_loader))
                print('batch :',batch)
                rank, mrr, hits = model.test(*batch, k=k)
                # rank, mrr, hits = test(test_data, model=model, device=device, batch_size=batch_size)
                print(f'Epoch: {epoch:03d}/{epochs}, Val Mean Rank: {rank:.2f}', f'Val MRR: {mrr:.4f}, Val Hits@{k}: {hits:.4f}')

        if use_wandb and epoch%eval_period == 0:
            wandb.log({"Val Mean Rank" : rank,
                               "Val MRR"       : mrr,
                               f"hits@{k}"       : hits,
                               "loss"          : loss,
                               "loss on test"  : test_loss})
            
        if use_wandb and epoch%eval_period != 0:
            wandb.log({"loss"         : loss,
                       "loss on test" : test_loss})

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
    
    return torch.Tensor(batch.apply(lambda row : lin_sim_on_mapped_terms(mapped_term1=row[0],
                                                                         mapped_term2=row[1],
                                                                         ),
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

class ComplEx(ComplEx):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        additionnal_negative_per_positive = 0
                ):
        super().__init__(num_nodes, num_relations, hidden_channels)

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = torch.nn.Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = torch.nn.Embedding(num_relations, hidden_channels, sparse=sparse)
        self.add_neg_per_pos = additionnal_negative_per_positive

    def make_pos_and_neg(self, head_index, rel_type, tail_index):
        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)
        for i in range(0,self.add_neg_per_pos):
                add_neg = self.random_sample(head_index, rel_type, tail_index)[3]
                print('aaan',add_neg[2])
                neg = torch.cat([neg,add_neg], dim=1)
                print('added additional negatives !')
        return pos, neg

class Random_ComplEx(ComplEx):
    def loss(
        self,
        head_index,
        rel_type,
        tail_index,
    ):

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = shuffle_tensor(torch.cat([pos_target, neg_target], dim=0))

        return F.binary_cross_entropy_with_logits(scores, target)

class tail_only_ComplEx(ComplEx):
    
    '''
    Overwritting random_sample() to make negative triples by setting a random tail to each triple,
    instead of setting a random head or tail.

    Moreover, test() is made on 1000 triples instead of all existing triples.

    I add a method to have different numbers of negative per positive when calculating the loss.

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

        tail_index = shuffle_tensor(tail_index.clone()).to(device)

        return head_index, rel_type, tail_index
    
    @torch.no_grad()
    def test(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        k: int = 10,
        log: bool = True,
    ):
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across 1000 tail entities.

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
         
        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []

        random_index = random.sample(range(tail_index.numel()), 1000)
        tested_triples_index = tqdm(random_index,
                                    desc=f"Calculating mean rank; MRR, Hit@10 among 1000 triples") if log else random_index
        
        # List the tails existing in the graph :
        tails = tail_index.unique().tolist()

        for i in tested_triples_index:

            h, r, t = head_index[i], rel_type[i], tail_index[i]

            # Select the index of 999 false tails :
            tail_indices = random.sample(tails, 999)
            # Add the true tail's index :
            tail_indices.append(i)
            # Score each triple (h, r, false_t) :
            ts = torch.tensor(tail_indices).to(device)
            rels = r.expand_as(ts).to(device)
            heads = h.expand_as(ts).to(device)
            self.to(device)
            scores = self(heads, rels, ts)
            # Sort the scores and find the rank of the true triplet score
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == 999).nonzero().item()

            # Using rank to precalculate metrics :
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

        # Calculate metrics :
        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k
    

    def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        '''

        # pos, neg = self.make_pos_and_neg(head_index=head_index,
        #                                  rel_type=rel_type,
        #                                  tail_index=tail_index)
        
        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)

        target = torch.cat([pos_target, neg_target], dim=0)
        # print(neg_target.size())


        return F.binary_cross_entropy_with_logits(scores, target)
    
class ComplEx_L_labels(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        Gaussian noise addition to labels.
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        lin_neg_target = lin_sims_for_batch(pos[2].to('cpu'), neg[2].to('cpu')).to(device)

        target = torch.cat([pos_target, lin_neg_target], dim=0)
        target = target

        return F.binary_cross_entropy_with_logits(scores, target)

class ComplEx_L_FRL_labels(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        Lin noise addition.
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        lin_neg_target = lin_sims_for_batch(pos[2].to('cpu'), neg[2].to('cpu')).to(device)

        target = torch.cat([pos_target, lin_neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, shuffle_tensor(target))
    
class ComplEx_LGN_labels(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        Gaussian noise addition to labels.
        '''
        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        lin_neg_target = lin_sims_for_batch(pos[2].to('cpu'), neg[2].to('cpu')).to(device)

        gn_target = torch.randn(size = scores.size()).to(device)
        target = torch.cat([pos_target, lin_neg_target], dim=0)
        print(target[0])
        target = target + gn_target
        print(target[0])
        print(target.size(),target)
        return F.binary_cross_entropy_with_logits(scores, target)

class ComplEx_FRL_labels(tail_only_ComplEx):
  
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        I randomised the full target tensor. The model should not learn.
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        lin_neg_target = lin_sims_for_batch(pos[2].to('cpu'), neg[2].to('cpu')).to(device)
        target = torch.cat([pos_target, lin_neg_target], dim=0)
        target = shuffle_tensor(target)

        return F.binary_cross_entropy_with_logits(scores, target)
  
class ComplEx_GN_labels(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        I randomised the full target tensor. The model should not learn.
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        lin_neg_target = lin_sims_for_batch(pos[2].to('cpu'), neg[2].to('cpu')).to(device)
        target = torch.cat([pos_target, lin_neg_target], dim=0)
        gn_target = torch.randn(size = scores.size()).to(device)

        return F.binary_cross_entropy_with_logits(scores, gn_target)

class ComplEx_BLS_labels(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a best linSim instead of label 0 on false tails.
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = best_lin_sims_for_batch(*neg).to(device)

        target = torch.cat([pos_target, neg_target], dim=0)
        # print(neg_target)

        return F.binary_cross_entropy_with_logits(scores, target)
  
class ComplEx_BLS_labels_more_neg(tail_only_ComplEx):
  def loss(
            self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a best linSim instead of label 0 on false tails.
        One add negatives. Should be used after e.g.:
        import play_wit_complex as pwc
        pwc.add_neg_per_pos = 10
        '''

        pos = head_index, rel_type, tail_index
        neg = self.random_sample(head_index, rel_type, tail_index)
        for i in range(0,self.add_neg_per_pos):
            add_neg = self.random_sample(head_index, rel_type, tail_index)
            neg = torch.cat([neg, add_neg], dim=0)
            
        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = best_lin_sims_for_batch(*neg).to(device)

        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)

class ComplEx_BLS_U_labels(tail_only_ComplEx):
  
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

class ComplEx_UBLS_labels(tail_only_ComplEx):
  
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
        target = target + lin_target

        return F.binary_cross_entropy_with_logits(scores, target)

class ComplEx_GN_U_labels(tail_only_ComplEx):
  
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
        neg_target = neg_target.to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        gn_target = torch.randn(size = scores.size()).to(device)
        print("usual",F.binary_cross_entropy_with_logits(scores, target))
        print("gn   ",F.binary_cross_entropy_with_logits(scores, gn_target))
        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, gn_target)
  
class ComplEx_UGN_labels(tail_only_ComplEx):
  
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
        neg_target = neg_target.to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        gn_target = torch.randn(size = scores.size()).to(device)
        target = target + gn_target

        return F.binary_cross_entropy_with_logits(scores, target) 

class ComplEx_RBLS_U_labels(tail_only_ComplEx):
  
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
  
class ComplEx_FRBLS_U_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        not-usual labels are shuffled.
        '''

        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        pos_target = torch.ones_like(pos_score).to(device)
        neg_target = torch.zeros_like(neg_score).to(device)
        lin_neg_target = best_lin_sims_for_batch(*neg).to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, shuffle_tensor(lin_target))
  
class ComplEx_FRL_U_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        not-usual labels are shuffled.
        '''

        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        pos_target = torch.ones_like(pos_score).to(device)
        neg_target = torch.zeros_like(neg_score).to(device)
        lin_neg_target = lin_sims_for_batch(pos[2].to('cpu'), neg[2].to('cpu')).to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, shuffle_tensor(lin_target))

class ComplEx_FRBLS_L_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have a linSim instead of label 0 on false tails.
        not-usual labels are shuffled.
        '''

        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        pos_score = self(*pos)
        neg_score = self(*neg)

        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        pos_target = torch.ones_like(pos_score).to(device)
        neg_target = torch.zeros_like(neg_score).to(device)
        lin_neg_target = best_lin_sims_for_batch(*neg).to(device)

        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, lin_target) + F.binary_cross_entropy_with_logits(scores, shuffle_tensor(lin_target))

class ComplEx_BLS_RBLS_U_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified usal labels, LinSim labels and randomised LinSin labels
        '''

        # Making + and - triples
        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        # Model predictions
        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        # Making Targets
        # + :
        pos_target = torch.ones_like(pos_score).to(device)
        # - :
        neg_target = torch.zeros_like(neg_score).to('cpu')
        lin_neg_target = best_lin_sims_for_batch(*neg).to(device)
        rand_lin_neg_target = shuffle_tensor(lin_neg_target.detach().clone()).to(device)
        neg_target = neg_target.to(device)
        # concat(+,-)
        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        lin_target = torch.cat([pos_target, lin_neg_target], dim=0).to(device)
        rand_lin_target = torch.cat([pos_target, rand_lin_neg_target], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, lin_target) + F.binary_cross_entropy_with_logits(scores, rand_lin_target)

class ComplEx_2BRLS_U_labels(tail_only_ComplEx):
  
  def loss(self,
            head_index: torch.Tensor,
            rel_type: torch.Tensor,
            tail_index: torch.Tensor,
            ) -> torch.Tensor:
            
        '''
        tail_only_ComplEx.loss() modified to have 2 terms on randomized Best LinSim labels.
        '''

        # Making + and - triples
        pos = head_index.to(device), rel_type.to(device), tail_index.to(device)
        neg = self.random_sample(head_index, rel_type, tail_index)

        # Model predictions
        pos_score = self(*pos)
        neg_score = self(*neg)
        scores = torch.cat([pos_score, neg_score], dim=0).to(device)

        # Making Targets
        # + :
        pos_target = torch.ones_like(pos_score).to(device)
        # - :
        neg_target = torch.zeros_like(neg_score).to('cpu')
        lin_neg_target = best_lin_sims_for_batch(*neg).to(device)
        rand_lin_neg_target1 = shuffle_tensor(lin_neg_target.detach().clone()).to(device)
        rand_lin_neg_target2 = shuffle_tensor(lin_neg_target.detach().clone()).to(device)

        neg_target = neg_target.to(device)
        # concat(+,-)
        target = torch.cat([pos_target, neg_target], dim=0).to(device)
        rand_lin_target1 = torch.cat([pos_target, rand_lin_neg_target1], dim=0).to(device)
        rand_lin_target2 = torch.cat([pos_target, rand_lin_neg_target2], dim=0).to(device)

        return F.binary_cross_entropy_with_logits(scores, target) + F.binary_cross_entropy_with_logits(scores, rand_lin_target1) + F.binary_cross_entropy_with_logits(scores, rand_lin_target2)

class ComplEx_with_normal_noise_and_usual_labels(ComplEx_BLS_U_labels):
  
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
        DEPRECATED : non-derivable additionnal loss term.
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
        DEPRECATED : non-derivable additionnal loss term.
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