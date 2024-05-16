import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import optimizer
import os

class BlackjackNNWithCount(nn.Module):
    def init(self):
        super(BlackjackNNWithCount, self).init()
        self.fc1 = nn.Linear(4, 128)  # 4 inputs: player hand, dealer's card, usable ace, card count
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)   # 1 output: probability of hitting

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
model = BlackjackNNWithCount()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

# get the data set
data = open( "data_sets/blackjack.data.3").readlines()
tags = open( "data_sets/blackjack.tags.3").readlines()
data_clean = []
tags_clean = []
#strip whitespace
first = True
for datum in data:
	if first:
		first = False
		continue
	clean_datum = datum[1:datum.index('\n')-1].strip().split(', ')
	clean_datum[0] = int( clean_datum[0] )
	clean_datum[1] = int( clean_datum[1] )
	print( clean_datum )
	data_clean = data_clean + [ clean_datum ]

first = True
for tag in tags:
	if first:
		first = False
		continue
	tag = tag[:tag.index('\n')]
	if tag == "h":
		tags_clean = tags_clean + [ 1.0 ]
	else:
		tags_clean = tags_clean + [ 0.0 ]

size = int( len(data)*(0.75) )

train_data = np.array( data_clean[1:size] )
train_tags = np.array( tags_clean[1:size] )
test_data = np.array( data_clean[size:] )
test_tags = np.array( tags_clean[size:] )


data_loader = train_data

for epoch in range(num_epochs):
    for game_states, decisions in data_loader:
        optimizer.zero_grad()
        outputs = model(game_states)
        loss = criterion(outputs, decisions)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss / len(data_loader)}")


    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


def train(learner, observations, actions, num_epochs=100):
    """Train function for learning a new policy using BC.
    
    Parameters:
        learner (Learner)
            A Learner object (policy)
        observations (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 11, ) 
        actions (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 3, )
        num_epochs (int)
            Number of epochs to run the train function for
    
    Returns:
        learner (Learner)
            A Learner object (policy)
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    dataset = TensorDataset(torch.tensor(observations, dtype = torch.float32), torch.tensor(actions, dtype = torch.float32)) # Create your dataset
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) # Create your dataloader
    
    # TODO: Complete the training loop here ###
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for obs_batch, action_batch in dataloader:
            optimizer.zero_grad()
            predicted_actions = learner(obs_batch)
            loss = loss_fn(predicted_actions, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    
    return learner


def get_checkpoint_path():
    """Return the path to save the best performing model checkpoint.
    
    Returns:
        checkpoint_path (str)
            The path to save the best performing model checkpoint
    """
    return 'best_model_checkpoint.pth'

class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
        

def create_loss_and_optimizer(model):
    """Create and return a loss function and optimizer.
    
    Parameters:
        model (torch.nn.Module)
            A neural network
        learning_rate (float)
            Learning rate for the optimizer
    
    Returns:
        loss_fn (function)
            The loss function for the model
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
    """
    lr=0.05
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    return loss_fn, optimizer

def train(x, y, model, loss_fn, optimizer, checkpoint_path, num_epochs=1000):
    """Train a model.
    
    Parameters:
        x (torch.Tensor)
            The input data
        y (torch.Tensor)
            The expected output data
        model (torch.nn.Module)
            A neural network
        loss_fn (function)
            The loss function
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
        checkpoint_path (str)
            The path to save the best performing checkpoint
        num_epochs (int)
            The number of epochs to train for
    
    Side Effects:
        - Save the best performing model checkpoint to `checkpoint_path`
    """
    best_loss = float('inf')

    for epoch in range(num_epochs):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), checkpoint_path)


def load_model_checkpoint(checkpoint_path):
    """Load a model checkpoint from disk.

    Parameters:
        checkpoint_path (str)
            The path to load the checkpoint from
    
    Returns:
        model (torch.nn.Module)
            The model loaded from the checkpoint
    """
    model = LinearRegression()
    model.load_state_dict(torch.load(checkpoint_path))
    return model