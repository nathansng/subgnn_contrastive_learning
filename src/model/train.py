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
Training, validation, and testing functions for the SubGNN & Contrastive Learning model
"""

# Train Contrastive Learning part of model 
def train_contrastive(model, loader, optimizer, loss_fn, p1=0.1, p2=0.2, device=None):
    # Set model to training
    model.train()
    
    # Run data through model and update model
    loss_all = 0
    n = 0 
    for data in loader: 
        data = data.to(device)
        optimizer.zero_grad()
        embeddings_1 = model(data, mode = "contrastive", p = p1)
        embeddings_2 = model(data, mode = "contrastive", p = p2)
        
        # Used as loss(embeddings, labels)
        loss = loss_fn(embeddings_1, embeddings_2)
        loss.backward()
        optimizer.step() 
        
        loss_all += data.num_graphs * loss.item()
        n += data.num_graphs
    return loss_all / n

# Evaluate Contrastive Learning part of model using graph embeddings 
def valid_contrastive(model, loader, loss_fn, p1=0.1, p2=0.2, device=None):
    # Set model to eval
    model.eval()
    
    with torch.no_grad():
        loss_all = 0
        n = 0
        for data in loader: 
            data = data.to(device)
            embeddings_1 = model(data, mode = "contrastive", p = p1)
            embeddings_2 = model(data, mode = "contrastive", p = p2)
            loss = loss_fn(embeddings_1, embeddings_2)
            
            loss_all += data.num_graphs * loss.item()
            n += data.num_graphs
    return loss_all / n

# Train prediction part of model 
def train_prediction(model, loader, optimizer, p=0.1, dropout=0.5, device=None):
    # Set model to training
    model.train()
    
    # Run data through model and update model 
    loss_all = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        results = model(data, mode = "prediction", p = p, dropout = dropout)
        loss = F.nll_loss(results, data.y)
    
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        n += len(data.y)
        optimizer.step()

    return loss_all / n

# Validate prediction part of model - Returns NLL loss between log likelihoods 
def valid_prediction(model, loader, p=0.1, dropout=0.5, device=None):
    # Set model to eval
    model.eval()
    
    # Run data through model
    with torch.no_grad():
        loss_all = 0
        n = 0
        for data in loader:
            data = data.to(device)
            results = model(data, mode = "prediction", p = p, dropout = dropout)
            loss = F.nll_loss(results, data.y)
                
            loss_all += data.num_graphs * loss.item()
            n += len(data.y)

    return loss_all / n

# Test prediction part of model - Returns accuracy of model 
def test_prediction(model, loader, p=0.1, dropout=0.5, device=None):
    # Set model to eval
    model.eval() 
    
    # Run data through model and make predictions
    with torch.no_grad():
        correct = 0
        for data in loader: 
            data = data.to(device)
            results = model(data, mode = "prediction", p = p, dropout = dropout)
            pred = results.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# Split data into k-Folds 
def separate_data(dataset_len, seed=0, n_splits=10):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_list = []
    for idx in folds.split(np.zeros(dataset_len), np.zeros(dataset_len)):
        idx_list.append(idx)
    return idx_list