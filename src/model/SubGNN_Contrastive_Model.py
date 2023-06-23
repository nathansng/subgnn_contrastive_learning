import os.path as osp
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from torch_geometric.data import DataLoader, Data
from torch_geometric.data.dataloader import Collater
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GINConv, GINEConv, global_add_pool

from pytorch_metric_learning import losses

"""
SubGNN & Contrastive Loss model 
"""

class SubGNN_Contrastive(nn.Module):
    def __init__(self, num_features, num_reps, num_classes, hidden_units, device, use_aux_loss=True):
        super(SubGNN_Contrastive, self).__init__()

        # Set starting parameters for model 
        self.num_features = num_features   # Number of initial features 
        self.num_reps = num_reps           # Number of features in representation vector 
        self.num_classes = num_classes     # Number of different classes
        self.dim = hidden_units            # Number of units for hidden layers
        self.use_aux_loss = use_aux_loss   # Whether to include aux loss to total loss
        
        self.device = device

        # Number of layers in model
        self.num_layers = 4

        self.convs = nn.ModuleList()        # Made of num_layers GINConv (linear -> batchnorm1d -> relu -> linear)
        self.bns = nn.ModuleList()          # Made of num_layers BatchNorm1d 
        self.reps = nn.ModuleList()         # Layer between base model and contrastive learning representation
        self.fcs = nn.ModuleList()          # Made of num_layers + 1 Linear layers mapping from num_features or dim to num_reps

        # Add initial layer from num_features to dim 
        self.convs.append(GINConv(nn.Sequential(nn.Linear(self.num_features, self.dim), nn.BatchNorm1d(self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))))
        self.bns.append(nn.BatchNorm1d(self.dim))
        self.reps.append(nn.Linear(self.num_features, self.num_reps))
        self.reps.append(nn.Linear(self.dim, self.num_reps))
        self.fcs.append(nn.Linear(self.num_features, self.num_classes))
        self.fcs.append(nn.Linear(self.dim, self.num_classes))

        # Add additional layers from dim to dim 
        for i in range(self.num_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(self.dim, self.dim), nn.BatchNorm1d(self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))))
            self.bns.append(nn.BatchNorm1d(self.dim))
            self.reps.append(nn.Linear(self.dim, self.num_reps))
            self.fcs.append(nn.Linear(self.dim, self.num_classes))
        
    def reset_parameters(self):
        # Resets parameters for Linear, GINConv, and BatchNorm1d layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
                
    def forward(self, data, mode="test", p=None, dropout=None, num_runs=20):
        # Runs different modes based on whether running contrastive loss or making predictions
        if mode == 'contrastive':
            return self.contrastive(data, p, num_runs)
        else:
            return self.prediction(data, p, dropout, num_runs)
        
    def contrastive(self, data, p, num_runs):
        # Trains contrastive model and representation vector model 
        
        # Note: num_runs in DropGNN is average number of nodes in each graph in dataset
        # Note: p is 2 * 1 / (1 + gamma), but for this project, p is selected to create augmented views 
        
        self.p = p
        self.num_runs = num_runs
        
        # Store all graphs in sampled batch as one large graph with separate components
        x = data.x                     # All nodes and their features (# nodes x # node features)
        edge_index = data.edge_index   # All edge index pairs from large single graph
        batch = data.batch             # Batch numbers that group nodes within the same graph with same batch number
        
        # Do runs in parallel by repeating nodes and creating num_runs different views
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()   # Creates num_runs copy of node features
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()   #  Randomly determine whether node is dropped within each copy of num_runs
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)  # Drop nodes from graphs  
        del drop
        
        # Allow gradients to update base model 
        if self.training:
            for layer in self.convs: 
                for p in layer.parameters():
                    p.requires_grad = True

            for layer in self.bns:
                for p in layer.parameters():
                    p.requires_grad = True
        
        # Run augmented subgraph through model 
        outs = [x]  # Used to store n-hop neighborhood representations, after running through model n times
        x = x.view(-1, x.size(-1))  # Concat all num_run copies of nodes 
        run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1) # Transform edge_index to correspond to the same nodes in concatenated form  
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)  # Run node features and edge indices through CONV layer 
            x = self.bns[i](x)  # Run resulting values through BatchNorm1d
            x = F.relu(x)   # Run final values through RELU
            outs.append(x.view(self.num_runs, -1, x.size(-1)))  # Return x back to original stacked form 
        del run_edge_index
        
        # Aggregates results of runs by taking mean of each run and summing results of runs
        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)                  # Take average of all node features of same nodes 
            x = global_add_pool(x, batch)      # Take the sum of all node features for nodes in same graph 
            x = self.reps[i](x)                # Run graph features into linear layer to get contrastive representation
            if out is None:
                out = x
            else:
                out += x
                
        # Returns all contrastive graph embeddings in batch 
        return out
    
    def prediction(self, data, p, dropout, num_runs):
        self.p = p
        self.dropout = dropout
        self.num_runs = num_runs
        
        # Create intermediate representations 
        x = data.x 
        edge_index = data.edge_index
        batch = data.batch 
        
        # Do runs in parallel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()   # Flattens features and creates num_runs copy of them 
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()   #  Returns a tensor of randomly dropped nodes based on p (p = probability of dropping) 
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)  # Drop nodes from data  
        del drop
        
        # Stop gradients from updating base model 
        for layer in self.convs:
            for p in layer.parameters():
                p.requires_grad = False
                
        for layer in self.bns:
            for p in layer.parameters():
                p.requires_grad = False
        
        # Run augmented subgraph through model 
        outs = [x]  # Used to store view of x after each layer 
        x = x.view(-1, x.size(-1))  # Swap dimensions of data features 
        run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1) # Expand edge_index and augment values
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)  # Run node features and edge indices through CONV layer 
            x = self.bns[i](x)  # Run resulting values through BatchNorm1d
            x = F.relu(x)   # Run final values through RELU
            outs.append(x.view(self.num_runs, -1, x.size(-1)))    # Rearrange dimensions and append to outs 
        del run_edge_index
        
        # Aggregates results of runs by summing mean and applying random dropout (not dropping out nodes)
        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x
        
        # Returns the likelihood of each outcome class
        return F.log_softmax(out, dim=-1) 