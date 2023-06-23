import sys
import json
import torch
from torch import nn
from torch import optim

from src.data.data import download_data
from src.model.SubGNN_Contrastive_Model import SubGNN_Contrastive
from src.model.train_loop import training_loop, evaluation_loop
from src.model.train import separate_data

def main(targets):
    # Read in config files 
    with open('config/config.json') as f:
        configs = json.load(f)
        
        dataset = configs["dataset"]
        model_config = configs["model_config"]
        train_config = configs["train_config"]
        n_splits = configs["n_splits"]
        
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")
        
    # Download data 
    dataset = download_data(dataset)
    
    # Update model configs based on dataset 
    model_config["num_features"] = dataset.num_features
    model_config["num_classes"] = dataset.num_classes
    model_config["device"] = device
    
    # Update training configs based on model 
    train_config["model"] = SubGNN_Contrastive(**model_config).to(device)
    train_config["dataset"] = dataset 
    train_config["splits"] = separate_data(len(dataset), seed=0, n_splits=n_splits)
    train_config["device"] = device
                           
    # Evaluate model 
    eval_results = evaluation_loop(**train_config)
    
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)