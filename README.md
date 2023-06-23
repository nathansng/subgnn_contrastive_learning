# SubGNN & Contrastive Learning

## SubGNN and Contrastive Learning Research Project

The goal of this project is to determine whether applying contrastive learning to subgraph neural networks improves the performance of graph classification models. Contrastive learning helps the model learn how to make better graph embeddings by learning similar embeddings for the same graphs and distinct embeddings for different graphs. Using the improved embeddings, graph prediction should ideally be easier, improving the performance in graph classification. The subgraph neural network used is inspired by DropGNN and modified to use contrastive learning and config files to modify the model parameters. Subgraphs are created by randomly dropping out nodes in the graph and then the resulting embeddings for each subgraph are aggregated to create the final graph embedding. 

## Downloading Data

The data used is the IMDB-BINARY and IMDB-MULTI, which are popular benchmark graph datasets. The datasets contain graphs representing movies, where actors/actresses represent nodes and edges exist between actors/actresses that act in the same movies. The datasets will be automatically downloaded when running the program. 

## SubGNN Contrastive Model

The SubGNN_Contrastive model has been implemented and allows users to use the mode for both contrastive learning and downstream graph predictions. The model inputs are nodes and node features, and edge ids that represent the edges between nodes. Using the contrastive learning portion, the model returns the learned graph embeddings. Using the graph prediction portion, the model returns the graph predictions. 

The tunable parameters of the model include: 

- Size of graph embeddings
- Size of hidden layers
- Number of layers
- Node dropout probabilities (tunable using the training configurations)
- Graph embedding dropout probability (tunable using the training configuration)

## Training Modules

This project also includes functions to train both the contrastive learning and graph prediction portions of the SubGNN Contrastive model. The tunable parameters for the automated training and evaluation loops include: 

- Batch size
- Epochs
- Learning rate
- Random seed
- Number of k-fold splits

## Running Code

*Note*: It is highly recommended to run the model on a GPU. Running the code on a GPU will make the program run significantly faster than only CPU. 

To run the code, run `python run.py`. The program will create a SubGNN Contrastive model using the parameters from the config file and evaluate the model. The results are both printed and saved to a text file for future reference. 

The tunable parameters of the model and training module can be edited in the `config.json` file in the config directory. 
