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

from .train import train_contrastive, valid_contrastive, train_prediction, valid_prediction, test_prediction

"""
Automates training process of SubGNN & Contrastive Learning model 
"""

# Train model - Final model is saved 
def training_loop(model, dataset, train_idx, test_idx, batch_size, epochs, p1, p2, p, dropout, device, lr=0.001, seed=0, m=10, filename=None):
    """
    Runs a single training loop based on given training and testing indices 
    """
    # Set random seeds 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set batch size and number of epochs 
    BATCH = batch_size
    NUM_EPOCHS = epochs 
    LR = lr
    
    # Create training and testing datasets
    train_dataset = dataset[train_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)*50/(len(train_dataset)/BATCH))), batch_size=BATCH, drop_last=False, collate_fn=Collater(follow_batch=[],exclude_keys=[]))
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    
    # Set up for contrastive learning
    loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss())   # Specify contrastive loss function to use 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)      # Optimizer for model to use 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # Used to adjust learning rate while training 
    
    # CONTRASTIVE LEARNING: Train model on contrastive representation 
    print("STARTING CONTRASTIVE LEARNING")
    if filename != None:
        with open(filename, "a") as f: 
            print("STARTING CONTRASTIVE LEARNING", file=f)
    
    contrastive_losses = []
    for epoch in range(NUM_EPOCHS):
        if epoch % m == 0:
            start = time.time()

        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train_contrastive(model, train_loader, optimizer, loss_func, p1=p1, p2=p2, device=device)
        scheduler.step()
        test_loss = valid_contrastive(model, test_loader, loss_func, p1=p1, p2=p2, device=device)
        contrastive_losses.append(test_loss)

        if epoch % m == 0:
            print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                'Val Loss: {:.7f}, Time: {:7f}'.format(
                    epoch, lr, train_loss, test_loss, time.time() - start), flush=True)
            if filename != None:
                with open(filename, "a") as f: 
                    print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                        'Val Loss: {:.7f}, Time: {:7f}'.format(
                            epoch, lr, train_loss, test_loss, time.time() - start), flush=True, file=f)
            
    # Set up for prediction 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)      # Optimizer for model to use 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # Used to adjust learning rate while training 
    
    # PREDICTION: Train model using contrastive representations to make predictions 
    print("\nSTARTING PREDICTION LEARNING")
    if filename != None: 
        with open(filename, "a") as f: 
            print("\nSTARTING PREDICTION LEARNING", file=f)
    
    prediction_losses = []
    for epoch in range(NUM_EPOCHS):
        if epoch % m == 0:
            start = time.time()

        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train_prediction(model, train_loader, optimizer, p=p, dropout=dropout, device=device)
        scheduler.step()
        test_loss = valid_prediction(model, test_loader, p=p, dropout=dropout, device=device)
        prediction_losses.append(test_loss)

        if epoch % m == 0:
            print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                'Val Loss: {:.7f}, Time: {:7f}'.format(
                    epoch, lr, train_loss, test_loss, time.time() - start), flush=True)
            if filename != None:
                with open(filename, "a") as f: 
                    print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                        'Val Loss: {:.7f}, Time: {:7f}'.format(
                            epoch, lr, train_loss, test_loss, time.time() - start), flush=True, file=f)
            
    # Test final accuracy of final model 
    test_acc = test_prediction(model, test_loader, dropout=dropout, device=device)
    print(f"\nFinal Prediction Accuracy: {test_acc}\n")
    if filename != None: 
        with open(filename, "a") as f: 
            print(f"\nFinal Prediction Accuracy: {test_acc}\n", file=f)
    
    return contrastive_losses, prediction_losses, test_acc


# Evaluate model - Model training is not saved 
def evaluation_loop(model, dataset, splits, batch_size, epochs, p1, p2, p, dropout, device, lr=0.001, seed=0, m=10, filename=None):
    # Train model on different splits, meant to evaluate model, not save best model
    contrastive_loss = []
    prediction_loss = []
    test_accuracies = []
    
    # Train a new model on every fold for evaluation
    for i, (train_idx, test_idx) in enumerate(splits): 
        print(f"Running Split {i}")
        if filename != None: 
            with open(filename, "a") as f: 
                print(f"Running Split {i}", file=f)
        
        model.reset_parameters()    # Resets upon every new fold 
        c_loss, p_loss, t_acc = training_loop(model, dataset, train_idx, test_idx, batch_size, epochs, p1, p2, p, dropout, device, lr, seed, m, filename)
        contrastive_loss.append(torch.tensor(c_loss))
        prediction_loss.append(torch.tensor(p_loss))
        test_accuracies.append(t_acc)
        
    # Calculate average contrastive loss and return best epoch for contrastive loss
    contrastive_loss = torch.stack(contrastive_loss, dim=0)
    contrastive_loss_mean = contrastive_loss.mean(dim=0)
    best_contrastive_epoch = contrastive_loss_mean.argmin()

    # Calculate average prediction loss and return best epoch for predictions 
    prediction_loss = torch.stack(prediction_loss, dim=0)
    prediction_loss_mean = prediction_loss.mean(dim=0)
    best_prediction_epoch = prediction_loss_mean.argmin()
    
    # Print average final prediction accuracy
    test_accuracies = torch.tensor(test_accuracies)
    print(f"Average Test Accuracy: {test_accuracies.mean()}\n")
    if filename != None: 
        with open(filename, "a") as f: 
            print(f"Average Test Accuracy: {test_accuracies.mean()}\n", file=f)
    
    return (contrastive_loss, contrastive_loss_mean, best_contrastive_epoch), (prediction_loss, prediction_loss_mean, best_prediction_epoch), test_accuracies