# %% [markdown]
# # LIBRARIES

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, save, load
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from sklearn.metrics import precision_score, recall_score, fbeta_score
from tabulate import tabulate
import copy
import torch.nn.init as init
from sklearn.linear_model import LinearRegression
from torch.optim.lr_scheduler import StepLR
from itertools import product
import matplotlib.patches as patches
import csv
import ast  # To safely evaluate strings as Python expressions
import re
import os
#!pip install torch_optimizer torchmetrics
import torch_optimizer as optim2

from torchmetrics.functional.regression import mean_absolute_percentage_error
from torch.autograd import Variable

from scipy.stats import norm


# %% [markdown]
# # DEVICE CONFIGURATION

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# %% [markdown]
# #CONTRACTIVE LOSS

# %%

mse_loss = nn.BCELoss(size_average = False)

lam = 1e-4

def CAE_loss(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss

    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.

    Contrastive loss plays a crucial role in maintaining the similarity
    and correlation of latent representations across different modalities.
    This is because it helps to ensure that similar instances are represented
    by similar vectors and dissimilar instances are represented by dissimilar vectors.


    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder

    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term

    Returns:
        Variable: the (scalar) CAE loss
    """
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


# %% [markdown]
# # FIRST ORDER NETWORK ARCHITECTURE DEFINITION

# %%
class FirstOrderNetwork(nn.Module):
    def __init__(self, hidden_units, data_factor, use_gelu):
        """
        Initializes the FirstOrderNetwork with specific configurations.

        Parameters:
        - hidden_units (int): The number of units in the hidden layer.
        - data_factor (int): Factor to scale the amount of data processed.
                             A factor of 1 indicates the default data amount,
                             while 10 indicates 10 times the default amount.
        - use_gelu (bool): Flag to use GELU (True) or ReLU (False) as the activation function.
        """
        super(FirstOrderNetwork, self).__init__()

        # Define the encoder, hidden, and decoder layers with specified units
        self.fc1 = nn.Linear( 48 , hidden_units , bias = False) # Encoder
        self.fc2 = nn.Linear( hidden_units, 48 , bias = False) # Decoder

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)

        # Set the data factor
        self.data_factor = data_factor

        # Initialize network weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights of the encoder, hidden, and decoder layers uniformly."""
        init.uniform_(self.fc1.weight, -1.0, 1.0)
        init.uniform_(self.fc2.weight, -1.0, 1.0)

    def encoder(self, x):
      h1 = self.dropout(self.relu(self.fc1(x)))
      return h1

    def decoder(self, z, prev_h2 , cascade_rate):
      # First apply the linear transformation
      h2 = self.fc2(z)

      # Apply sigmoid to every 6-unit subpart of each pattern
      for i in range(0, h2.size(1), bits_per_letter):
          h2[:, i:i+bits_per_letter] = self.sigmoid(h2[:, i:i+bits_per_letter])

      #cascade mode
      if prev_h2 is not None:
        h2 = cascade_rate * h2 + (1 - cascade_rate) * prev_h2

      return h2

    def forward(self, x, prev_h1, prev_h2 , cascade_rate):


      """
      Defines the forward pass through the network.

      Parameters:
      - x (Tensor): The input tensor to the network.

      Returns:
      - Tensor: The output of the network after passing through the layers and activations.
      """
      h1 = self.encoder(x)
      h2 = self.decoder(h1, prev_h2, cascade_rate)

      return h1 , h2


# %% [markdown]
# # SECOND ORDER NETWORK ARCHITECTURE DEFINITION

# %%
class SecondOrderNetwork(nn.Module):
    def __init__(self, use_gelu, hidden_second):
        super(SecondOrderNetwork, self).__init__()
        # Define a linear layer for comparing the difference between input and output of the first-order network
        self.comparison_layer = nn.Linear(48, hidden_second)

        # Linear layer for determining wagers, mapping from 100 features to a single output
        self.wager = nn.Linear(hidden_second, 1)

        # Dropout layer to prevent overfitting by randomly setting input units to 0 with a probability of 0.5 during training
        self.dropout = nn.Dropout(0.5)

        # Select activation function based on the `use_gelu` flag
        if use_gelu:
          self.activation=torch.nn.GELU()
        else:
          self.activation=torch.nn.ReLU()

        # Additional activation functions for potential use in network operations
        self.sigmoid = torch.sigmoid

        self.softmax = nn.Softmax()

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Uniformly initialize weights for the comparison and wager layers
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, first_order_input, first_order_output, prev_comparison, cascade_rate):
        # Calculate the difference between the first-order input and output
        comparison_matrix = first_order_input - first_order_output

        # Pass the difference through the comparison layer and apply the chosen activation function
        comparison_out=self.dropout(self.activation(self.comparison_layer(comparison_matrix)))
        if prev_comparison is not None:
          comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        # Calculate the wager value, applying dropout and sigmoid activation to the output of the wager layer
        wager = self.sigmoid(self.wager(comparison_out))

        return wager , comparison_out

# %% [markdown]
# # FUNCTIONS THAT CREATES RANDOM , GRAMMAR A, AND GRAMMAR B WORDS
# 
# 

# %%
#GRAMMAR A AND GRAMMAR B WORDS ARE BUILD BASED ON THE ARCHITECTURE SHOWN ON FIGURE 5.1, FROM THE PAPER "Five Transfer of implicit knowledge across domains: How
#implicit and how abstract?" (DIENES, 1997)

#mechanism to generate a random word with the dominion of the allowed letter within grammar A and B
def Generate_Word_Random():
    grammar_word = ""
    number_letters = random.randint(3, 8)
    allowed_letters = ["x", "v", "m", "t", "r"]  # Letters from Grammar A and B
    while len(grammar_word) < number_letters:
        current_letter = random.choice(allowed_letters)
        grammar_word += current_letter
    return grammar_word

#mechanim to generate a grammar A word based on the grammar A architecture of the Dienes, 1997
def Generate_Grammar_A():
  grammar_A_word=""
  number_letters=random.randint(3,8)
  position=1
  i=0
  while len(grammar_A_word) < number_letters:

    current_path = random.randint(1, 2)

    (grammar_A_word := grammar_A_word + "x", position := 2) if (position, current_path) == (1, 1) else (grammar_A_word := grammar_A_word + "v", position := 3) if (position, current_path) == (1, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_A_word)==number_letters:
      break
    (grammar_A_word := grammar_A_word + "m", position := 2) if (position, current_path) == (2, 1) else (grammar_A_word := grammar_A_word + "x", position := 4) if (position, current_path) == (2, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_A_word)==number_letters:
      break
    (grammar_A_word := grammar_A_word + "t", position := 3) if (position, current_path) == (3, 1) else (grammar_A_word := grammar_A_word + "v", position := 5) if (position, current_path) == (3, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_A_word)==number_letters:
      break
    (grammar_A_word := grammar_A_word + "t", position := 4) if (position, current_path) == (4, 1) else (grammar_A_word := grammar_A_word + "m", position := 6) if (position, current_path) == (4, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_A_word)==number_letters:
      break
    (grammar_A_word := grammar_A_word + "r", position := 3) if (position, current_path) == (5, 1) else (grammar_A_word := grammar_A_word + "m", position := 6) if (position, current_path) == (5, 2) else None

    if position==6:
      break
  return grammar_A_word

#mechanim to generate a grammar B word based on the grammar B architecture of the Dienes, 1997

def Generate_Grammar_B():
  grammar_B_word=""
  number_letters=random.randint(3,8)
  position=1
  i=0
  while len(grammar_B_word) < number_letters:

    current_path = random.randint(1, 2)

    (grammar_B_word := grammar_B_word + "x", position := 2) if (position, current_path) == (1, 1) else (grammar_B_word := grammar_B_word + "v", position := 3) if (position, current_path) == (1, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_B_word)==number_letters:
      break
    (grammar_B_word := grammar_B_word + "x", position := 5) if (position, current_path) == (2, 1) else (grammar_B_word := grammar_B_word + "m", position := 3) if (position, current_path) == (2, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_B_word)==number_letters:
      break
    (grammar_B_word := grammar_B_word + "v", position := 4) if (position, current_path) == (3, 1) else (grammar_B_word := grammar_B_word + "t", position := 5) if (position, current_path) == (3, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_B_word)==number_letters:
      break
    (grammar_B_word := grammar_B_word + "t", position := 4) if (position, current_path) == (4, 1) else (grammar_B_word := grammar_B_word + "r", position := 2) if (position, current_path) == (4, 2) else None

    current_path = random.randint(1, 2)
    if len(grammar_B_word)==number_letters:
      break

    if position==5:
       if current_path==1:
        grammar_B_word+="r"
       elif len(grammar_B_word)>2:
          break

  return grammar_B_word

# %% [markdown]
# #FUNCTION THAT ENCODES THE WORD TO GIVE TO THE NEURAL NETWORK
# 
# 

# %%
# each letter encoded as a unique one-hot vector. Since there are 5 possible letters, each letter could be represented as a 5-dimensional vector (e.g., 'a' = [1, 0, 0, 0, 0], 'b' = [0, 1, 0, 0, 0], etc.).
#A string with a maximum of 8 letters would then be a 40-dimensional vector (8 letters x 5 dimensions per letter).
def encode_word(word):
    # Define the mapping
    mapping = {"x": [1, 0, 0, 0, 0, 0],
               "v": [0, 1, 0, 0, 0, 0],
               "m": [0, 0, 1, 0, 0, 0],
               "t": [0, 0, 0, 1, 0, 0],
               "r": [0, 0, 0, 0, 1, 0]}

    # Initialize the output array with zeros
    encoded = [0] * 48  # 48 elements, all zeros

    # Encode each letter and place it in the output array
    for i, letter in enumerate(word):
        start_index = i * bits_per_letter
        end_index = start_index + bits_per_letter

        # Ensure we don't exceed the 48 elements limit
        if end_index > 48:
            break

        encoded[start_index:end_index] = mapping.get(letter, [0] * bits_per_letter)

    return encoded

# %% [markdown]
# # FUNCTION THAT MAKES ARRAYS OF WORDS (for pre-training, training, testing)
# 
# 

# %%
#A functiion that makes an array of multiple encoded grammar words, be random, grammar A, or grammar B, returns as tensors
def Array_Words(grammar_type , number, output=False):
  list_words=[]
  while len(list_words) < number:

    if grammar_type==1:
      generated= Generate_Word_Random()
    if grammar_type==2:
      generated= Generate_Grammar_A()
    if grammar_type==3:
      generated= Generate_Grammar_B()
    generated_encoded= encode_word(generated)

    """
    if output:
      print(generated)
      print(generated_encoded)
    """

    list_words.append(generated_encoded)

  list_words=torch.Tensor(list_words).to(device)
  return list_words

# %% [markdown]
# # FUNCTION THAT RETURNS SECOND ORDER TARGETS
# 

# %%
#A function that uses input and output of first order networks to return the targets of high wager(1.0) or low wager(0.0), to be used in the training
def target_second(input, output):
    if input.shape != output.shape:
        raise ValueError("Input and output must have the same shape")

    num_rows, num_cols = input.shape
    result = torch.zeros(num_rows)

    for i in range(num_rows):
        # Count the number of 1s in the input row
        input_indexes = (input[i] == 1).nonzero(as_tuple=True)[0]
        num_ones = input_indexes.size(0)

        # Get the indexes of the top x values in the output row
        _, output_indexes = torch.topk(output[i], num_ones)

        # Compare and set the result
        if set(input_indexes.tolist()) == set(output_indexes.tolist()):
            result[i] = 1.0

    wager = torch.Tensor(result).to(device)
    return wager

# %% [markdown]
# #PRECISION
# 

# %%
#evaluation function of utility of the training. Calculates precision based    , True_positives /  ( True_positives +  False_positives)


def calculate_metrics(patterns_tensor, output_first_order):
    # Initialize counters for true positives (tp), false positives (fp),
    # true negatives (tn), and false negatives (fn)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Create the predicted_patterns list
    predicted_patterns = []

    for pattern in output_first_order:
        predicted_pattern = torch.zeros_like(pattern)
        for i in range(0, len(pattern), bits_per_letter):
            pack = pattern[i:i+bits_per_letter]
            max_index = torch.argmax(pack)

            if pattern[i + max_index] > 0.1:
                predicted_pattern[i + max_index] = 1

        predicted_patterns.append(predicted_pattern)

    # Convert predicted_patterns list to a tensor
    predicted_patterns_tensor = torch.stack(predicted_patterns)

    # Calculate metrics
    for i in range(len(patterns_tensor)):
        tp += (patterns_tensor[i] * predicted_patterns_tensor[i]).sum().item()
        fp += ((1 - patterns_tensor[i]) * predicted_patterns_tensor[i]).sum().item()
        tn += ((1 - patterns_tensor[i]) * (1 - predicted_patterns_tensor[i])).sum().item()
        fn += (patterns_tensor[i] * (1 - predicted_patterns_tensor[i])).sum().item()

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision



# %% [markdown]
# # Function that performs linear regression

# %%
#calculates linear approximation function to graph on the plots

def perform_linear_regression(epoch_list, precision):
    # Perform linear regression
    X = np.array(epoch_list).reshape(-1, 1)
    y = np.array(precision)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    return y_pred

# %% [markdown]
# #AVERAGE, MAX, MIN FUNCTIONS FOR PLOTS

# %%
#calculate averge over the trained networks over epochs

def calculate_average(network_data, start_network, end_network):
    num_epochs = len(network_data[0])
    average = [0] * num_epochs
    num_networks = end_network - start_network
    for epoch in range(num_epochs):
        sum_loss = sum(network_data[i][epoch] for i in range(start_network, end_network))
        average[epoch] = sum_loss / num_networks
    return average

#calculate max reached over the trained networks over epochs

def calculate_max(network_data, start_network, end_network):
    num_epochs = len(network_data[0])
    maximum = [0] * num_epochs
    for epoch in range(num_epochs):
        maximum[epoch] = max(network_data[i][epoch] for i in range(start_network, end_network))
    return maximum

#calculate max reached over the trained networks over epochs

def calculate_min(network_data, start_network, end_network):
    num_epochs = len(network_data[0])
    minimum = [float('inf')] * num_epochs
    for epoch in range(num_epochs):
        minimum[epoch] = min(network_data[i][epoch] for i in range(start_network, end_network))
    return minimum

# %% [markdown]
# #ASSIGNMENT OF FIRST AND SECOND ORDER NETWORK, AND DEFINITION OF CRITERIONS
# 

# %%
#define the architecture, optimizers, loss functions, and schedulers for pre training

def prepare_pre_training(hidden, hidden_second,factor,gelu,stepsize, gam):
  first_order_network = FirstOrderNetwork(hidden,factor,gelu).to(device)
  second_order_network = SecondOrderNetwork(gelu, hidden_second).to(device)

  #Binary Cross-Entropy Loss: especially used in classification tasks, either correctly classified it or not, or correctly high wager or not
  criterion_1 = CAE_loss
  criterion_2 = nn.BCELoss(size_average = False)

  #Allow different optimizers to be used
  if optimizer_name == 'ADAM':
    optimizer_1 = optim.Adam(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Adam(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'ADAMAX':
    optimizer_1 = optim.Adamax(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Adamax(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'RADAM':
    optimizer_1 = optim2.RAdam(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim2.RAdam(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'RANGER':
    optimizer_1 = optim2.Ranger(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim2.Ranger(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'RANGERQH':
    optimizer_1 = optim2.RangerQH(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim2.RangerQH(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'RANGERVA':
    optimizer_1 = optim2.RangerVA(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim2.RangerVA(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'SGD':
    optimizer_1 = optim.SGD(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.SGD(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'ADAGRAD':
    optimizer_1 = optim.Adagrad(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Adagrad(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'ADAMW':
    optimizer_1 = optim.AdamW(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.AdamW(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'RMSPROP':
    optimizer_1 = optim.RMSprop(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.RMSprop(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'ADADELTA':
    optimizer_1 = optim.Adadelta(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Adadelta(second_order_network.parameters(), lr=learning_rate_2)
  elif optimizer_name == 'RPROP':
    optimizer_1 = optim.Rprop(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Rprop(second_order_network.parameters(), lr=learning_rate_2)


  # Learning rate schedulers
  scheduler_1 = StepLR(optimizer_1, step_size=stepsize, gamma=gam)
  scheduler_2 = StepLR(optimizer_2, step_size=stepsize, gamma=gam)

  #to later restore the weights of the 1st order network
  initial_first_order_weights = copy.deepcopy(first_order_network.state_dict())  # Store initial weights

  return first_order_network, second_order_network, criterion_1, criterion_2, optimizer_1, optimizer_2, scheduler_1, scheduler_2, initial_first_order_weights

# %% [markdown]
# # PRE TRAINING LOOP

# %%
def get_num_args(func):
    return func.__code__.co_argcount

def pre_train(first_order_network, second_order_network, criterion_1, criterion_2, optimizer_1, optimizer_2, scheduler_1, scheduler_2, initial_first_order_weights, factor , meta, cascade_rate, type_cascade):

  precision= np.zeros((n_epochs_pre))
  if type_cascade ==1:
    cascade_rate_one=cascade_rate
    cascade_iterations_one= int(1.0/cascade_rate)
    cascade_rate_two= cascade_rate
    cascade_iterations_two= int(1.0/cascade_rate)
  elif type_cascade==2:
    cascade_rate_one= cascade_rate
    cascade_iterations_one= int(1.0/cascade_rate)
    cascade_rate_two=1.0
    cascade_iterations_two= 1
  elif type_cascade==3:
    cascade_rate_one= 1.0
    cascade_iterations_one= 1
    cascade_rate_two= cascade_rate
    cascade_iterations_two= int(1.0/cascade_rate)
  else:
    cascade_rate_one= 1.0
    cascade_iterations_one= 1
    cascade_rate_two= 1.0
    cascade_iterations_two= 1
      
  number_patterns=int(patterns_number_pre*factor)
  
  for epoch in range(n_epochs_pre):
      #print('Epoch {}/{}'.format(epoch, n_epochs_pre) )

      #generation online of patterns every epoch, and pass over the networks
      if epoch==0:
        patterns_tensor = Array_Words(1, number_patterns, True)

      patterns_tensor = Array_Words(1, number_patterns)

      # Forward pass through the first-order network
      #hidden_representation , output_first_order = first_order_network(patterns_tensor)
      hidden_representation=None
      output_first_order=None
      comparison_out=None

      for i in range(cascade_iterations_one):
        # Forward pass through the first-order network
        hidden_representation , output_first_order = first_order_network(patterns_tensor, hidden_representation, output_first_order, cascade_rate_one)


      patterns_tensor=patterns_tensor.requires_grad_(True)
      output_first_order=output_first_order.requires_grad_(True)

      optimizer_1.zero_grad()

      # Conditionally execute the second-order network pass and related operations
      if meta:

          for i in range(cascade_iterations_two):
            # Forward pass through the second-order network with inputs from the first-order network
            output_second_order , comparison_out = second_order_network(patterns_tensor, output_first_order, comparison_out, cascade_rate_two)

          output_second_order = output_second_order.squeeze()
          output_second_order=output_second_order.requires_grad_(True)
          # Forward pass through the second-order network with inputs from the first-order network
          #output_second_order = second_order_network(patterns_tensor, output_first_order).squeeze()

          order_2_tensor_without = target_second(patterns_tensor, output_first_order)

          # Calculate the loss for the second-order network (wagering decision based on comparison)
          loss_2 = criterion_2(output_second_order, order_2_tensor_without).requires_grad_()

          # Backpropagate the second-order network's loss
          loss_2.backward(retain_graph=True)  # Allows further backpropagation for loss_1 after loss_2

          # Update second-order network weights
          optimizer_2.step()

          scheduler_2.step()

          optimizer_2.zero_grad()

          epoch_2_order[epoch] = loss_2.item()
          
      else:
          # Skip computations for the second-order network
          with torch.no_grad():

              # Potentially forward pass through the second-order network without tracking gradients
              for i in range(cascade_iterations_two):
                output_second_order , comparison_out = second_order_network(patterns_tensor, output_first_order, comparison_out, cascade_rate_two)
              output_second_order = output_second_order.squeeze()

      # Calculate the loss for the first-order network (accuracy of stimulus representation)

      num_args = get_num_args(criterion_1)

      """
      print(output_first_order.shape)
      print(patterns_tensor.shape)
      print(output_first_order[0])
      print(patterns_tensor[0])
      """

      if num_args == 2:
        loss_1 = criterion_1(  output_first_order , patterns_tensor )

      else:
        W = first_order_network.state_dict()['fc1.weight']
        loss_1 = criterion_1( W, patterns_tensor, output_first_order, hidden_representation, lam )

      # Backpropagate the first-order network's loss
      loss_1.backward()

      # Update first-order network weights
      optimizer_1.step()

      # Update the first-order scheduler
      scheduler_1.step()

      epoch_1_order[epoch] = loss_1.item()

      precision[epoch] = calculate_metrics(patterns_tensor, output_first_order)
      """
      if epoch == n_epochs_pre - 1:

        print(patterns_tensor.shape)
        print(output_first_order.shape)
        print(patterns_tensor[0])
        print(output_first_order[0])
      """


  first_order_network.load_state_dict(initial_first_order_weights) #reload initial 1st order weights

  return first_order_network , second_order_network , epoch_1_order , epoch_2_order, precision

# %% [markdown]
# # PRE-TRAINING PLOTS

# %%
def pre_train_plots(epoch_1_order, epoch_2_order, precision , title):

  # LOSS PLOTS

  # Set up the plot with 1 row and 2 columns
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Adjust the figure size as needed

  # First graph for 1st Order Network
  ax1.plot(epoch_list, epoch_1_order, linestyle='--', marker='o', color='g')
  ax1.legend(['1st Order Network'])
  ax1.set_xlabel('Number of EPOCH - PRETRAINING PHASE')
  ax1.set_ylabel('LOSS')

  # Second graph for 2nd Order Network
  ax2.plot(epoch_list, epoch_2_order, linestyle='--', marker='o', color='b')
  ax2.legend(['2nd Order Network'])
  ax2.set_xlabel('Number of EPOCH - PRETRAINING PHASE')
  ax2.set_ylabel('LOSS')


  plt.suptitle(title, fontsize=16)

  # Display the plots side by side
  plt.tight_layout()
  plt.savefig('ArtificialGrammar_Pre_training_Loss_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  plt.show()
  plt.close(fig)

  #PRECISION PLOTS
  fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # Adjust the figure size as needed

  ax.plot(epoch_list, precision, linestyle='--', marker='o', color='r', label='Precision')
  ax.set_xlabel('Number of EPOCH - PRETRAINING PHASE')
  ax.set_ylabel('Precision')

  # Perform linear regression
  y_pred = perform_linear_regression(epoch_list, precision)
  # Plot the linear function
  ax.plot(epoch_list, y_pred, linestyle='-', color='b', label='Linear Fit')

  # Display the legend
  ax.legend()

  plt.suptitle(title, fontsize=16)

  # Display the plot
  plt.tight_layout()
  plt.savefig('ArtificialGrammar_Pre_training_Precision_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  plt.show()
  plt.close(fig)

# %% [markdown]
# # MODEL LOADING FOR TRAINING AND SET UP OF THE 30 NETWORKS

# %%
def create_networks(first_order_network , second_order_network,hidden, hidden_second,factor,gelu,stepsize, gam):
  #SAVING THE MODELS
  PATH = './cnn1.pth'
  PATH_2 = './cnn2.pth'

  #save the weights of the pretrained networks
  torch.save(first_order_network.state_dict(), PATH)
  torch.save(second_order_network.state_dict(), PATH_2)

  networks = []

  for i in range(num_networks):
      loaded_model_trai = FirstOrderNetwork(hidden,factor,gelu)
      loaded_model_2_trai = SecondOrderNetwork(gelu, hidden_second)

      loaded_model_trai.load_state_dict(torch.load(PATH))
      loaded_model_2_trai.load_state_dict(torch.load(PATH_2))

      loaded_model_trai.to(device)
      loaded_model_2_trai.to(device)

      criterion_1 = CAE_loss
      criterion_2 = nn.BCELoss(size_average = False)
      
      if optimizer_name == 'ADAM':
        optimizer_1 = optim.Adam(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.Adam(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'ADAMAX':
        optimizer_1 = optim.Adamax(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.Adamax(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'RADAM':
        optimizer_1 = optim2.RAdam(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim2.RAdam(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'RANGER':
        optimizer_1 = optim2.Ranger(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim2.Ranger(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'RANGERQH':
        optimizer_1 = optim2.RangerQH(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim2.RangerQH(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'RANGERVA':
        optimizer_1 = optim2.RangerVA(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim2.RangerVA(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'SGD':
        optimizer_1 = optim.SGD(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.SGD(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'RMSPROP':
        optimizer_1 = optim.RMSprop(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.RMSprop(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'ADAGRAD':
        optimizer_1 = optim.Adagrad(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.Adagrad(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'ADAMW':
        optimizer_1 = optim.AdamW(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.AdamW(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name =='ADADELTA':
        optimizer_1 = optim.Adadelta(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.Adadelta(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      elif optimizer_name == 'RPROP':
        optimizer_1 = optim.Rprop(loaded_model_trai.parameters(), lr=learning_rate_training_1)
        optimizer_2 = optim.Rprop(loaded_model_2_trai.parameters(), lr=learning_rate_training_2)
        
      # Learning rate schedulers
      scheduler_1 = StepLR(optimizer_1, step_size=stepsize, gamma=gam)
      scheduler_2 = StepLR(optimizer_2, step_size=stepsize, gamma=gam)

      networks.append([loaded_model_trai, loaded_model_2_trai , criterion_1 , criterion_2 , optimizer_1 , optimizer_2 , scheduler_1, scheduler_2])

  return networks

# %% [markdown]
# # TRAINING
# 
# 
# 
# 
# 

# %%
# %% 
def training(networks, n_epochs_train, start_network, end_network, factor, meta, cascade_rate, type_cascade):
    # Set the number of training epochs
    n_epochs_tra_1 = n_epochs_train
    # Create a list of epoch numbers for tracking
    epoch_list = list(range(1, n_epochs_tra_1 + 1))
    
    # Initialize arrays to store loss and precision metrics for each network across epochs
    epoch_1_order = np.zeros((len(networks), n_epochs_tra_1))   # First-order network losses
    epoch_2_order = np.zeros((len(networks), n_epochs_tra_1))   # Second-order network losses
    precision_high = np.zeros((len(networks), n_epochs_tra_1))  # Precision metrics
    
    # Configure cascade rates based on the specified type
    if type_cascade == 1:
        # Both networks use the same cascade rate
        cascade_rate_one = cascade_rate
        cascade_iterations_one = int(1.0/cascade_rate)
        cascade_rate_two = cascade_rate
        cascade_iterations_two = int(1.0/cascade_rate)
    elif type_cascade == 2:
        # Only first network uses cascade, second network processes in one step
        cascade_rate_one = cascade_rate
        cascade_iterations_one = int(1.0/cascade_rate)
        cascade_rate_two = 1.0
        cascade_iterations_two = 1
    elif type_cascade == 3:
        # First network processes in one step, only second network uses cascade
        cascade_rate_one = 1.0
        cascade_iterations_one = 1
        cascade_rate_two = cascade_rate
        cascade_iterations_two = int(1.0/cascade_rate)
    else:
        # No cascade for either network (default)
        cascade_rate_one = 1.0
        cascade_iterations_one = 1
        cascade_rate_two = 1.0
        cascade_iterations_two = 1
    
    patterns_number=45*factor
    # Loop through the specified range of networks
    for index in range(start_network, end_network):
        # For each epoch in the training process
        for epoch in range(n_epochs_tra_1):
            # Generate input data patterns for training
            patterns_tensor = Array_Words(2, patterns_number)
            
            # Initialize hidden representation and outputs to None
            hidden_representation = None
            output_first_order = None
            comparison_out = None
            
            # Process through the first-order network with cascade
            for i in range(cascade_iterations_one):
                # Forward pass through the first-order network
                # Gradually accumulate activation with each iteration
                hidden_representation, output_first_order = networks[index][0](
                    patterns_tensor, hidden_representation, output_first_order, cascade_rate_one)
            
            # Enable gradient tracking for inputs and outputs
            patterns_tensor = patterns_tensor.requires_grad_(True)
            output_first_order = output_first_order.requires_grad_(True)
            
            # Reset gradients for the first-order network optimizer
            networks[index][4].zero_grad()
            
            # Process through the second-order network (metacognitive component)
            meta = False  # Override meta flag (seems inconsistent with parameter)
            if meta == True:
                # Meta-cognitive training path
                for i in range(cascade_iterations_two):
                    # Forward pass through the second-order network
                    output_second_order, comparison_out = networks[index][1](
                        patterns_tensor, output_first_order, comparison_out, cascade_rate_two)
                
                # Process outputs
                comparison_out = comparison_out.squeeze()
                output_second_order = output_second_order.squeeze()
                output_second_order = output_second_order.requires_grad_(True)
                
                # Calculate target for second-order and compute loss
                order_2_tensor = target_second(patterns_tensor, output_first_order)
                loss_2 = networks[index][3](output_second_order, order_2_tensor).requires_grad_()
                
                # Backpropagation for second-order network
                loss_2.backward(retain_graph=True)
                networks[index][5].step()  # Update second-order network weights
                networks[index][7].step()  # Update additional network weights
                networks[index][5].zero_grad()  # Reset gradients
            else:
                # Standard (non-meta) training path
                for i in range(cascade_iterations_two):
                    # Forward pass through the second-order network
                    output_second_order, comparison_out = networks[index][1](
                        patterns_tensor, output_first_order, comparison_out, cascade_rate_two)
                
                # Process outputs
                comparison_out = comparison_out.squeeze()
                output_second_order = output_second_order.squeeze()
                
                # Calculate target for second-order and compute loss
                order_2_tensor = target_second(patterns_tensor, output_first_order)
                loss_2 = networks[index][3](output_second_order, order_2_tensor)
            
            # Compute first-order loss based on available loss function
            num_args = get_num_args(networks[index][2])
            if num_args == 2:
                # Simple loss function with two arguments
                loss_1 = networks[index][2](output_first_order, patterns_tensor)
            else:
                # Complex loss function with additional parameters
                W = networks[index][0].state_dict()['fc1.weight']
                loss_1 = networks[index][2](W, patterns_tensor, output_first_order,
                                          hidden_representation, lam)
            
            # Backpropagation for first-order network
            loss_1.backward()
            networks[index][4].step()  # Update first-order network weights
            networks[index][6].step()  # Update additional network weights
            
            # Store metrics for this epoch and network
            epoch_1_order[index][epoch] = loss_1.detach().item()
            epoch_2_order[index][epoch] = loss_2.detach().item()
            precision_high[index][epoch] = calculate_metrics(patterns_tensor, output_first_order)
    
    # Flatten the loss and precision arrays for easier analysis
    flattened_loss_1 = [item for sublist in epoch_1_order for item in sublist]
    flattened_precision = [item for sublist in precision_high for item in sublist]
    
    # Return the updated networks and all collected metrics
    return networks, epoch_list, epoch_1_order, epoch_2_order, precision_high, flattened_loss_1, flattened_precision

def training_plots(epoch_list, epoch_1_order, epoch_2_order,precision_high,start_network, end_network ,title):
  epoch_list_high=epoch_list
  avg_1_order = calculate_average(epoch_1_order, start_network, end_network)
  max_1_order = calculate_max(epoch_1_order, start_network, end_network)
  min_1_order = calculate_min(epoch_1_order, start_network, end_network)

  avg_2_order = calculate_average(epoch_2_order, start_network, end_network)
  max_2_order = calculate_max(epoch_2_order, start_network, end_network)
  min_2_order = calculate_min(epoch_2_order, start_network, end_network)

  avg_precision_high=calculate_average(precision_high, start_network, end_network)
  max_precision_high=calculate_max(precision_high, start_network, end_network)
  min_precision_high=calculate_min(precision_high, start_network, end_network)


  # Set up the plot with 1 row and 2 columns
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Adjust the figure size as needed

  # First graph for 1st Order Network
  ax1.plot(epoch_list, avg_1_order,  linestyle='--', marker='o', color='b' , label='Average Loss - 1st Order')
  ax1.fill_between(epoch_list, min_1_order, max_1_order, color='b', alpha=0.2)
  ax1.legend(['Average loss 1st Order Network'])
  ax1.set_ylabel('LOSS')
  # Second graph for 2nd Order Network
  ax2.plot(epoch_list, avg_2_order,  linestyle='--', marker='o', color='r' , label='Average Loss - 2nd Order')
  ax2.fill_between(epoch_list, min_2_order, max_2_order, color='r', alpha=0.2)
  ax2.legend(['Average loss 2nd Order Network'])
  ax2.set_ylabel('LOSS')

  plt.suptitle(title, fontsize=16)

  # Display the plots side by side
  if start_network==0:
    ax1.set_xlabel('Number of EPOCH - TESTING PHASE (high consciousness)')
    ax2.set_xlabel('Number of EPOCH - TESTING PHASE (high consciousness)')
    plt.tight_layout()
    plt.savefig('ArtificialGrammar_training_Loss_High_Consciousness_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  else:
    ax1.set_xlabel('Number of EPOCH - TESTING PHASE (low consciousness)')
    ax2.set_xlabel('Number of EPOCH - TESTING PHASE (low consciousness)')
    plt.tight_layout()
    plt.savefig('ArtificialGrammar_training_Loss_Low_Consciousness_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')

  plt.show()
  plt.close(fig)

  # PLOT PRECISION, Linear Regression
  fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
  ax.plot(epoch_list, avg_precision_high, linestyle='--', marker='o', color='r', label='Average Precision')
  ax.fill_between(epoch_list, min_precision_high, max_precision_high, color='r', alpha=0.2, label='Precision Range')
  y_pred_linear = perform_linear_regression(epoch_list, avg_precision_high)
  ax.plot(epoch_list, y_pred_linear, linestyle='-', color='b', label='Linear Fit')
  ax.set_ylabel('Precision')

  plt.suptitle(title, fontsize=16)

  if start_network==0:
    ax.set_xlabel('Number of EPOCH - TESTING PHASE (high consciousness)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('ArtificialGrammar_training_Precision_High_Consciousness_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  else:
    ax.set_xlabel('Number of EPOCH - TESTING PHASE (low consciousness)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('ArtificialGrammar_training_Precision_Low_Consciousness_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  plt.show()
  plt.close(fig)

# %% [markdown]
# #SCATTER PLOT - TRAINING PHASE

# %%
#plots performance vs 1st order loss of the n trained networks, both high consciousness and low consicousness networks
def training_scatter_plot(flattened_loss_1_high,flattened_loss_1_low, flattened_precision_high,flattened_precision_low,title):
  fig = plt.figure()

  plt.scatter(flattened_loss_1_low, flattened_precision_low, edgecolor='red', facecolors='red', label='Low Consciousness')
  plt.scatter(flattened_loss_1_high, flattened_precision_high, edgecolor='blue', facecolors='none',
              marker='o', label='High Consciousness')

  y_pred_linear = perform_linear_regression(flattened_loss_1_high, flattened_precision_high)
  plt.plot(flattened_loss_1_high, y_pred_linear, linestyle='-', color='b', label='Linear Fit - High Consciousness')

  y_pred_linear = perform_linear_regression(flattened_loss_1_low, flattened_precision_low)
  plt.plot(flattened_loss_1_low, y_pred_linear, linestyle='-', color='r', label='Linear Fit - Low Consciousness')

  plt.xlabel('Loss 1st order Network')
  plt.ylabel('Precision')
  plt.title('Loss vs Precision - TRAINING PHASE')
  plt.grid(True)

  plt.legend()

  plt.suptitle(title, fontsize=16)

  plt.tight_layout()
  plt.savefig('ArtificialGrammar_training_Scatter_plot_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  plt.show()
  plt.close(fig)

# %% [markdown]
# # TESTING LOOP

# %%
def compute_metrics(TP, TN, FP, FN):
    """Compute precision, recall, F1 score, and accuracy."""
    precision = round(TP / (TP + FP), 2) if (TP + FP) > 0 else 0
    recall = round(TP / (TP + FN), 2) if (TP + FN) > 0 else 0
    f1_score = round(2 * (precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0
    accuracy = round((TP + TN) / (TP + TN + FP + FN), 2) if (TP + TN + FP + FN) > 0 else 0
    return precision, recall, f1_score, accuracy

def testing(networks, factor, cascade_rate,seeds, type_cascade):
  #Model loading and evaluation set up
  for network in range(len(networks)):
    networks[network][0].eval()
    networks[network][1].eval()

  patterns_number=int(60*factor)
  Random_baseline_patterns= Array_Words(1, patterns_number)
  
  Random_baseline_patterns = torch.Tensor(Random_baseline_patterns).to(device)

  numbers_networks = list(range(1, num_networks*seeds +1))
  loss_1_networks= np.zeros(num_networks)
  loss_2_networks= np.zeros(num_networks)
  precision_networks= np.zeros(num_networks)
  accuracy_2nd_order = np.zeros(num_networks)
  f1_2nd_order = np.zeros(num_networks)
  recall_2nd_order = np.zeros(num_networks)
  precision_2nd_order = np.zeros(num_networks)

  random_baseline= np.zeros(num_networks)

  if type_cascade ==1:
    cascade_rate_one=cascade_rate
    cascade_iterations_one= int(1.0/cascade_rate)
    cascade_rate_two= cascade_rate
    cascade_iterations_two= int(1.0/cascade_rate)
  elif type_cascade==2:
    cascade_rate_one= cascade_rate
    cascade_iterations_one= int(1.0/cascade_rate)
    cascade_rate_two=1.0
    cascade_iterations_two= 1
  elif type_cascade==3:
    cascade_rate_one= 1.0
    cascade_iterations_one= 1
    cascade_rate_two= cascade_rate
    cascade_iterations_two= int(1.0/cascade_rate)
  else:
    cascade_rate_one= 1.0
    cascade_iterations_one= 1
    cascade_rate_two= 1.0
    cascade_iterations_two= 1
      
  for network in range(len(networks)):


    with torch.no_grad():
      Testing_grammar_A= Array_Words(2, int(len(networks) *factor))
      Testing_grammar_B= Array_Words(3, int(len(networks)*factor) )

      Testing_patterns = torch.cat((Testing_grammar_A, Testing_grammar_B), dim=0)
      Testing_patterns= torch.Tensor(Testing_patterns).to(device)

      patterns_tensor= Testing_patterns

      hidden_representation=None
      output_first_order=None
      comparison_out=None

      for i in range(cascade_iterations_one):
        # Forward pass through the first-order network
        hidden_representation , output_first_order = networks[network][0](patterns_tensor, hidden_representation, output_first_order, cascade_rate_one)

      for i in range(cascade_iterations_two):
        # Forward pass through the second-order network with inputs from the first-order network
        output_second_order , comparison_out = networks[network][1](patterns_tensor, output_first_order, comparison_out, cascade_rate_two)

      output_second_order = output_second_order.squeeze()

      order_2_tensor = target_second(patterns_tensor , output_first_order)

      loss_2 = networks[network][3](   output_second_order  , order_2_tensor     ).requires_grad_()

      num_args = get_num_args(networks[network][2])

      if num_args == 2:
        loss_1 = networks[network][2](  output_first_order , patterns_tensor )
      else:
        W = networks[network][0].state_dict()['fc1.weight']
        loss_1 = networks[network][2]( W, patterns_tensor, output_first_order,
                            hidden_representation, lam )

      loss_1_networks[network] = loss_1
      loss_2_networks[network] = loss_2

      precision_networks[network]=  calculate_metrics(patterns_tensor, output_first_order)
      random_baseline[network]=  calculate_metrics(patterns_tensor, Random_baseline_patterns)


      # Calculate accuracy for the second-order network
      wagers = output_second_order.cpu().numpy().flatten()
      targets_2 = target_second(patterns_tensor, output_first_order).cpu().numpy().flatten()

      # Convert targets to binary classification
      targets_2 = (targets_2 > 0).astype(int)

      # Calculate True Positives, True Negatives, False Positives, and False Negatives
      TP = np.sum((wagers > threshold) & (targets_2 > threshold))
      TN = np.sum((wagers <= threshold) & (targets_2 <= threshold))
      
      FP = np.sum((wagers > threshold) & (targets_2 <= threshold))
      FN = np.sum((wagers <= threshold) & (targets_2 > threshold))

      # Compute precision, recall, F1 score, and accuracy for the second-order network
      precision_h, recall_h, f1_score_h, accuracy_h = compute_metrics(TP, TN, FP, FN)
      
      
      precision_2nd_order[network] = precision_h
      recall_2nd_order[network] = recall_h
      f1_2nd_order[network] = f1_score_h
      accuracy_2nd_order[network] = accuracy_h



  max_value_high = np.max(precision_networks[0:num_networks//2])
  max_value_low = np.max(precision_networks[num_networks//2:num_networks])
  mean_value=np.mean(precision_networks)

  #print(accuracy_2nd_order)

  return loss_1_networks, loss_2_networks, precision_networks, precision_2nd_order, recall_2nd_order, f1_2nd_order, accuracy_2nd_order, random_baseline, numbers_networks, max_value_high, max_value_low, mean_value

# %% [markdown]
# #PLOTS(TABLE AND SCATTER PLOT) - TESTING PHASE

# %%
def metrics_testing_table(metric):
  metric_avg_high=np.mean(metric[0:len(metric)//2])
  metric_avg_low=np.mean(metric[len(metric)//2:])
  metric_std_high=np.std(metric[0:len(metric)//2])
  metric_std_low=np.std(metric[len(metric)//2:])

  return metric_avg_high, metric_avg_low, metric_std_high, metric_std_low

def plots_testing(random_baseline, precision_networks, precision_2nd_order, recall_2nd_order, f1_2nd_order, accuracy_2nd_order, loss_1_networks, loss_2_networks, numbers_networks , title):
  
  metric_2nd_order = accuracy_2nd_order
  
  #TABLE LOSS AND PRECISION OF NETWORKS AFTER TESTING

  mean_random_baseline = np.mean(random_baseline)
  
  loss_1_networks = np.array(loss_1_networks)
  loss_2_networks = np.array(loss_2_networks)

  # Find the index of the lowest precision value
  max_precision_index = np.argmax(precision_networks)
  min_loss_1_index = np.argmin(loss_1_networks)
  min_loss_2_index = np.argmin(loss_2_networks)

  # Set up the plot
  fig, ax = plt.subplots(figsize=(7, 10))
  ax.axis('tight')
  ax.axis('off')
  table_data = []

  # Add header row with different background color
  header_row = ['Network', 'Loss 1', 'Loss 2', 'Precision']
  cell_colors = ['lightyellow', 'lightgray', 'lightgray', 'lightgray']
  table_data.append(header_row)
  

  for i, (number, loss_1, loss_2, precision_value) in enumerate(zip(numbers_networks, loss_1_networks, loss_2_networks, precision_networks)):
      row_data = [number, f'{loss_1:.3f}', f'{loss_2:.3f}', f'{precision_value:.3f}']
      table_data.append(row_data)

  # Create the table object
  table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', cellColours=[cell_colors]*len(table_data))

  # Set background color for title1 cells
  for i, title1 in enumerate(header_row):
      table[0, i].set_facecolor('lightblue')

  for i, title1 in enumerate(header_row):
      table[max_precision_index + 1, i].set_facecolor('lightgreen')

  for i, title1 in enumerate(header_row):
      table[min_loss_1_index + 1, i].set_facecolor('lightgreen')

  for i, title1 in enumerate(header_row):
      table[min_loss_2_index + 1, i].set_facecolor('lightgreen')

  table.auto_set_font_size(False)
  table.set_fontsize(12)
  plt.tight_layout()

  plt.savefig('ArtificialGrammar_testing_table_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  plt.show()
  plt.close(fig)


  fig = plt.figure()
  std_random_baseline = np.std(random_baseline)
  upper_bound = mean_random_baseline + std_random_baseline
  lower_bound = mean_random_baseline - std_random_baseline

  split_index = len(precision_networks) // 2

  flattened_precision_high = precision_networks[:split_index]
  flattened_precision_low = precision_networks[split_index:]
  flattened_loss_1_high = loss_1_networks[:split_index]
  flattened_loss_1_low = loss_1_networks[split_index:]


  plt.scatter(flattened_loss_1_low, flattened_precision_low, edgecolor='red', facecolors='red', label='Low Consciousness')
  plt.scatter(flattened_loss_1_high, flattened_precision_high, edgecolor='blue', facecolors='none',
              marker='o', label='High Consciousness')

  y_pred_linear = perform_linear_regression(flattened_loss_1_high, flattened_precision_high)
  plt.plot(flattened_loss_1_high, y_pred_linear, linestyle='-', color='b', label='Linear Fit - High Consciousness')

  y_pred_linear = perform_linear_regression(flattened_loss_1_low, flattened_precision_low)
  plt.plot(flattened_loss_1_low, y_pred_linear, linestyle='-', color='r', label='Linear Fit - Low Consciousness')

  plt.axhline(y=mean_random_baseline, color='black', linestyle='--', label='Random Baseline')
  plt.fill_between( (0.2, 1.0), lower_bound, upper_bound, color='black', alpha=0.2)

  plt.xlabel('Loss 1st order Network')
  plt.ylabel('Precision')
  plt.grid(True)

  plt.legend()

  plt.tight_layout()
  plt.suptitle(title, fontsize=16, y=1.1)

  plt.savefig('testing_scatter_plot_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
  plt.show()
  plt.close(fig)



  # Calculate the metrics for high and low consciousness
  precision_1_high, precision_1_low, precision_std_high, precision_std_low = metrics_testing_table(precision_networks)
  accuracy_high, accuracy_low, accuracy_std_high, accuracy_std_low = metrics_testing_table(metric_2nd_order)

  # Prepare the data for the table
  table_data = [
      ["Precision (1st Order)", f"{precision_1_high:.2f} ± {precision_std_high:.2f}", f"{precision_1_low:.2f} ± {precision_std_low:.2f}"],
      ["Accuracy (2nd Order)", f"{accuracy_high:.2f} ± {accuracy_std_high:.2f}", f"{accuracy_low:.2f} ± {accuracy_std_low:.2f}"]
  ]

  # Extract metric values for color scaling
  metric_values_high = np.array([precision_1_high, accuracy_high])
  metric_values_low = np.array([precision_1_low, accuracy_low])
  max_value = np.max([metric_values_high, metric_values_low])

  # Normalize the values for color scaling
  colors_high = metric_values_high / max_value
  colors_low = metric_values_low / max_value

  # Set up the table with colored cells
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.axis('off')
  ax.axis('tight')

  col_labels = ["Metric", "High Consciousness", "Low Consciousness"]
  row_labels = ["Precision (1st Order)", "Accuracy (2nd Order)"]

  # Initialize cell colors, defaulting to white for non-metric cells
  cell_colors = [["white"] * 3 for _ in range(len(table_data))]
  for i in range(len(table_data)):
      cell_colors[i][1] = plt.cm.RdYlGn(colors_high[i])
      cell_colors[i][2] = plt.cm.RdYlGn(colors_low[i])

  # Create the table object
  table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', rowLabels=row_labels, cellColours=cell_colors)
  table.auto_set_font_size(False)
  table.set_fontsize(10)
  table.scale(1.2, 1.2)

  plt.tight_layout()
  plt.suptitle(f"{title} - Metrics Summary", fontsize=16, y=1.02)

  plt.savefig(f'metrics_summary_{title.replace(" ", "_").replace("/", "_")}.png', bbox_inches='tight')
  plt.show()
  plt.close(fig)





# %% [markdown]
# #MAIN CODE - TRAINING DEFINITION

# %%
def train(hidden, hidden_second, factor, gelu, stepsize, gam, meta, cascade_rate, seeds, type_cascade, optimizer='RANGERVA'):
    """
    Main training function that orchestrates pre-training, training, and testing of the neural network system.
    
    Parameters:
    - hidden: Size of the hidden layer
    - factor: Scale factor for input patterns
    - gelu: Boolean flag indicating whether to use GELU activation function
    - stepsize: Learning rate step size
    - gam: Gamma parameter for scheduler
    - meta: Boolean flag for metacognitive training
    - cascade_rate: Rate for cascade model (controls gradual activation accumulation)
    - seeds: Number of random seeds to use for multiple training runs
    - type_cascade: Type of cascade model to use (1-4)
    - optimizer: Optimizer type (default: 'RANGERVA')
    
    Returns:
    - Various statistics, trained networks, and plot data
    """
    # Initialize lists to store results across multiple seed runs
    # For violin plots - comparing high vs low consciousness networks
    list_violin_1st = [[], []]  # First-order network precision for high/low consciousness
    list_violin_2nd = [[], []]  # Second-order network accuracy for high/low consciousness

    # For mean and standard deviation metrics
    list_1st_mean = []  # Mean precision for first-order network
    list_2nd_mean = []  # Mean accuracy for second-order network
    list_1st_std = []   # Standard deviation of precision for first-order network
    list_2nd_std = []   # Standard deviation of accuracy for second-order network
    
    # Testing metrics arrays - structure: [high_consciousness, low_consciousness, combined]
    precision_1st_testing = [[], [], []]  # Precision metrics for first-order network
    precision_2nd_testing = [[], [], []]  # Precision metrics for second-order network
    accuracy_2nd_testing = [[], [], []]   # Accuracy metrics for second-order network
    recall_2nd_testing = [[], [], []]     # Recall metrics for second-order network
    f1_2nd_testing = [[], [], []]         # F1 score metrics for second-order network
    
    # Random baseline and loss values across networks
    random_baseline_testing = []          # Random baseline performance for comparison
    loss_1_networks_testing = [[], [], []] # First-order network loss for testing
    loss_2_networks_testing = [[], [], []] # Second-order network loss for testing

    # Run training multiple times with different random seeds
    for seed in range(seeds):
        # Initialize global parameters for this seed run
        initialize_global(optimizer)

        # Prepare networks for pre-training phase
        # Creates first and second order networks with specified parameters
        first_order_network, second_order_network, criterion_1, criterion_2, optimizer_1, optimizer_2, scheduler_1, scheduler_2, initial_first_order_weights = prepare_pre_training(hidden, hidden_second, factor, gelu, stepsize, gam)

        # Pre-training phase - trains networks on random patterns
        # For AGL tasks, uses random strings to establish initial weights
        first_order_network_pre, second_order_network_pre, epoch_1_order_pre, epoch_2_order_pre, precision = pre_train(
            first_order_network, second_order_network, criterion_1, criterion_2, 
            optimizer_1, optimizer_2, scheduler_1, scheduler_2, 
            initial_first_order_weights, factor, meta, cascade_rate, type_cascade)

        # Create a set of networks for training, initialized with pre-trained weights
        networks = create_networks(first_order_network_pre, second_order_network_pre, hidden, hidden_second, factor, gelu, stepsize, gam)

        # High consciousness training phase
        # In the context of AGL, this would use grammar A patterns
        networks_high, epoch_list_high, epoch_1_order_high, epoch_2_order_high, precision_high, flattened_loss_1_high, flattened_precision_high = training(
            networks, num_training_high, 0, num_networks//2, factor, meta, cascade_rate, type_cascade)

        # Low consciousness training phase
        # Continues training the networks with potentially different parameters
        networks_low, epoch_list_low, epoch_1_order_low, epoch_2_order_low, precision_low, flattened_loss_1_low, flattened_precision_low = training(
            networks_high, num_training_low, num_networks//2, num_networks, factor, meta, cascade_rate, type_cascade)

        # Testing phase - evaluates network performance
        # For AGL, uses a mix of grammar A and grammar B strings
        loss_1_networks_t, loss_2_networks_t, precision_networks_testing, precision_2nd_order, recall_2nd_order, f1_2nd_order, accuracy_2nd_order, random_baseline, numbers_networks_testing, max_value_high_testing, max_value_low_testing, mean_value_testing = testing(
            networks_low, factor, cascade_rate, seeds, type_cascade)

        # Calculate metrics from testing results
        precision_1_high, precision_1_low, precision_std_high, precision_std_low = metrics_testing_table(precision_networks_testing)
        accuracy_high, accuracy_low, accuracy_std_high, accuracy_std_low = metrics_testing_table(accuracy_2nd_order)

        # Organize metrics for easier processing
        list_1s_metric = [precision_1_high, precision_1_low]
        list_2s_metric = [accuracy_high, accuracy_low]
        list_1s_metric_std = [precision_std_high, precision_std_low]
        list_2s_metric_std = [accuracy_std_high, accuracy_std_low]

        # Store metrics for this seed run
        list_1st_mean.append(list_1s_metric)
        list_2nd_mean.append(list_2s_metric)
        list_1st_std.append(list_1s_metric_std)
        list_2nd_std.append(list_2s_metric_std)
        
        # Extend testing results lists with data from this seed run
        # Split data between high consciousness networks (first half) and low consciousness networks (second half)
        precision_1st_testing[0].extend(precision_networks_testing[0:num_networks//2])
        precision_1st_testing[1].extend(precision_networks_testing[num_networks//2:])
        
        precision_2nd_testing[0].extend(precision_2nd_order[0:num_networks//2])
        precision_2nd_testing[1].extend(precision_2nd_order[num_networks//2:])
        
        accuracy_2nd_testing[0].extend(accuracy_2nd_order[0:num_networks//2])
        accuracy_2nd_testing[1].extend(accuracy_2nd_order[num_networks//2:])
        
        recall_2nd_testing[0].extend(recall_2nd_order[0:num_networks//2])
        recall_2nd_testing[1].extend(recall_2nd_order[num_networks//2:])
        
        f1_2nd_testing[0].extend(f1_2nd_order[0:num_networks//2])
        f1_2nd_testing[1].extend(f1_2nd_order[num_networks//2:])  
        
        # Convert loss arrays to lists before extending
        loss_2_networks_testing[0].extend(loss_2_networks_t[0:num_networks//2].tolist())
        loss_2_networks_testing[1].extend(loss_2_networks_t[num_networks//2:].tolist())
        
        loss_1_networks_testing[0].extend(loss_1_networks_t[0:num_networks//2].tolist())
        loss_1_networks_testing[1].extend(loss_1_networks_t[num_networks//2:].tolist())

        # Data for violin plots comparing high vs low consciousness networks
        list_violin_1st[0].extend(precision_networks_testing[0:num_networks//2])
        list_violin_1st[1].extend(precision_networks_testing[num_networks//2:])

        list_violin_2nd[0].extend(accuracy_2nd_order[0:num_networks//2])
        list_violin_2nd[1].extend(accuracy_2nd_order[num_networks//2:])
        
        # Collect random baseline results
        random_baseline_testing.extend(random_baseline)

    # Calculate averages across all seed runs
    avg_1st = np.mean(list_1st_mean, axis=0)  # Average precision for first-order network
    avg_2nd = np.mean(list_2nd_mean, axis=0)  # Average accuracy for second-order network
    std_1st = np.mean(list_1st_std, axis=0)   # Average std dev for first-order precision
    std_2nd = np.mean(list_2nd_std, axis=0)   # Average std dev for second-order accuracy
    
    # Combine high and low consciousness results for comprehensive analysis
    precision_1st_testing[2].extend(precision_1st_testing[0])
    precision_1st_testing[2].extend(precision_1st_testing[1])
    
    precision_2nd_testing[2].extend(precision_2nd_testing[0])
    precision_2nd_testing[2].extend(precision_2nd_testing[1])
    
    accuracy_2nd_testing[2].extend(accuracy_2nd_testing[0])
    accuracy_2nd_testing[2].extend(accuracy_2nd_testing[1])
    
    recall_2nd_testing[2].extend(recall_2nd_testing[0])
    recall_2nd_testing[2].extend(recall_2nd_testing[1])
    
    f1_2nd_testing[2].extend(f1_2nd_testing[0])
    f1_2nd_testing[2].extend(f1_2nd_testing[1])
    
    loss_2_networks_testing[2].extend(loss_2_networks_testing[0])
    loss_2_networks_testing[2].extend(loss_2_networks_testing[1])
    
    loss_1_networks_testing[2].extend(loss_1_networks_testing[0])
    loss_1_networks_testing[2].extend(loss_1_networks_testing[1])
      
    # Collect all data for visualization plots
    plot_data = {
        'pre_train': (epoch_1_order_pre, epoch_2_order_pre, precision),  # Pre-training metrics
        'training_high': (epoch_list_high, epoch_1_order_high, epoch_2_order_high, precision_high),  # High consciousness training
        'training_low': (epoch_list_low, epoch_1_order_low, epoch_2_order_low, precision_low),  # Low consciousness training
        'training_scatter': (flattened_loss_1_high, flattened_loss_1_low, flattened_precision_high, flattened_precision_low),  # Loss vs precision
        'testing': (random_baseline_testing, precision_1st_testing[2], precision_2nd_testing[2], 
                   recall_2nd_testing[2], f1_2nd_testing[2], accuracy_2nd_testing[2],  
                   loss_1_networks_testing[2], loss_2_networks_testing[2], numbers_networks_testing)  # Testing metrics
    }
    
    # Return comprehensive results data
    return avg_1st, avg_2nd, std_1st, std_2nd, max_value_high_testing, max_value_low_testing, mean_value_testing, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd
  
def plots(plot_data , title):
  #plot the results of a trained and tested network
  pre_train_plots(*plot_data['pre_train'],title)
  training_plots(*plot_data['training_high'], 0, num_networks//2,title)
  training_plots(*plot_data['training_low'], num_networks//2, num_networks,title)
  training_scatter_plot(*plot_data['training_scatter'],title)
  plots_testing(*plot_data['testing'],title)

def plot_scaling_data(plot_data):
    # This function should performance of a network when we scale the data several factors
    factors = plot_data['factors']
    max_values_high = plot_data['max_values_high']
    max_values_low = plot_data['max_values_low']
    mean_values = plot_data['mean_values']
    mean_random_baseline=[]
    upper_bound=[]
    lower_bound=[]
    for i in range(len(plot_data['random_baseline'])):
      mean=np.mean(plot_data['random_baseline'][i])
      std=np.std(plot_data['random_baseline'][i])
      mean_random_baseline.append(mean)
      upper_bound.append(mean+std)
      lower_bound.append(mean-std)

    fig = plt.figure()
    plt.figure()
    plt.plot(factors, max_values_high, linestyle='--', marker='o' , color='yellow',label='Max Value - high consciousness')
    plt.plot(factors, max_values_low, linestyle='--', marker='o' , color='blue',label='Max Value - low consciousness')
    plt.plot(factors, mean_values, linestyle='--', marker='o', color='red', label='Mean Value')

    print(mean_random_baseline)
    
    plt.plot(factors, mean_random_baseline,  linestyle='--', marker='o', color='black' , label='Random Baseline')
    plt.fill_between(factors, lower_bound , upper_bound, color='black', alpha=0.2)
    plt.xlabel('Data Scale Factor')
    plt.ylabel('Value')
    plt.title('Performance Metrics vs Data Scale Factor')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ArtificialGrammar_performance_metrics_vs_data_scale.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %%
def initialize_global(optimizer):
  global Input_Size_1 , Output_Size_1, Input_Size_2, learning_rate_1, learning_rate_2, learning_rate_training_1, learning_rate_training_2 , n_epochs_pre, momentum
  global temperature, patterns_number_pre, colors, threshold, bits_per_letter, n_epochs_pre
  global epoch_list, epoch_1_order, epoch_2_order, num_networks, num_training_high, num_training_low
  global optimizer_name
  
  optimizer_name = optimizer

  num_networks = 20 #default is 30
  num_training_high=12 #default is 12
  num_training_low=3 #default is 3
  n_epochs_pre = 30 #default is 60

  #NETWORKS SIZES
  Input_Size_1= 48
  Output_Size_1=48
  Input_Size_2= 48
  # pre-training and hyperparamentes

  learning_rate_1 = 0.4
  learning_rate_2 = 0.1

  learning_rate_training_1 = 0.4
  learning_rate_training_2 = 0.1

  momentum = 0.5
  temperature = 1.0
  threshold=0.5
  bits_per_letter=6   #6 bits to represent 1 letter, used in winner takes all

  patterns_number_pre = 80
  colors = [
      'b', 'g', 'r', 'c', 'm', 'y', 'k',
      '#FF5733', '#33FF57', '#5733FF', '#33FFFF', '#FF33FF', '#FFFF33',
      '#990000', '#009900', '#000099', '#999900', '#990099', '#009999',
      '#CC0000', '#00CC00', '#0000CC', '#CCCC00', '#CC00CC', '#00CCCC',
      '#FF6666', '#66FF66', '#6666FF', '#FFFF66', '#FF66FF', '#66FFFF'
  ]

  epoch_list = list(range(1, n_epochs_pre + 1))
  epoch_1_order= np.zeros(n_epochs_pre)
  epoch_2_order= np.zeros(n_epochs_pre)

def title(string):
    # Split the string into multiple lines if necessary
    words = string.split()
    max_words_per_line = max(1, len(words) // 3)  # Ensure at least one word per line

    lines = [' '.join(words[i:i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)]

    # Plot the title of the currently trained model, inside a rectangle
    fig, ax = plt.subplots()
    rectangle = patches.Rectangle((0.05, 0.1), 0.9, 0.4, linewidth=1, edgecolor='r', facecolor='blue', alpha=0.5)
    ax.add_patch(rectangle)

    # Dynamically calculate y positions
    y_positions = np.linspace(0.4, 0.2, len(lines))

    # Position the text over multiple lines
    for i, line in enumerate(lines):
        plt.text(0.5, y_positions[i], line, horizontalalignment='center', verticalalignment='center', fontsize=26, color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()
    plt.close(fig)



# Plotting function for scaling
def plot_scaling(scaling_metric, f1_scores_list, discr_perf_list, std_wager, std_discrimination, f1_scores_list_meta, discr_perf_list_meta, std_wager_meta, std_discrimination_meta,  x_axis):
    

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Transpose lists to plot correctly
    f1_scores_array = np.array(f1_scores_list).T
    discr_perf_array = np.array(discr_perf_list).T
    std_wager_array = np.array(std_wager).T
    std_discrimination_array = np.array(std_discrimination).T

   # Transpose lists to plot correctly
    f1_scores_array_meta = np.array(f1_scores_list_meta).T
    discr_perf_array_meta = np.array(discr_perf_list_meta).T
    std_wager_array_meta = np.array(std_wager_meta).T
    std_discrimination_array_meta = np.array(std_discrimination_meta).T
    
    scenarios = ['High consciousness', 'Low consciousness']


    # Plot F1 Scores
    for i, scenario in enumerate(scenarios):
        axes[0].plot(scaling_metric, f1_scores_array[i], label=scenario)
        axes[0].fill_between(scaling_metric, f1_scores_array[i] - std_wager_array[i], f1_scores_array[i] + std_wager_array[i], alpha=0.2)
        
    scenarios_meta = ['High consciousness(Maps)', 'Low consciousness(Maps)']

    # Plot F1 Scores
    for i, scenario in enumerate(scenarios_meta):
        axes[0].plot(scaling_metric, f1_scores_array_meta[i], label=scenario)
        axes[0].fill_between(scaling_metric, f1_scores_array_meta[i] - std_wager_array_meta[i], f1_scores_array_meta[i] + std_wager_array_meta[i], alpha=0.2)
        
    '''
   # Set the Y-axis range for the first subplot
    f1_first_value = f1_scores_array[0][0]
    f1_stdv = np.std(f1_scores_array)
    axes[0].set_ylim([f1_first_value - 5 * f1_stdv, f1_first_value + 5 * f1_stdv])
    '''
    axes[0].set_ylim([0, 1])


    axes[0].set_title('Accuracy 2nd Order Network',fontsize=28)
    axes[0].set_xlabel(x_axis,fontsize=24)
    axes[0].set_ylabel('Accuracy Score',fontsize=24)
    axes[0].legend(fontsize=16)
    axes[0].set_xscale('log')  # Set x-axis to log scale
    

    # Plot Discrimination Performances
    for i, scenario in enumerate(scenarios):
        axes[1].plot(scaling_metric, discr_perf_array[i], label=scenario)
        axes[1].fill_between(scaling_metric, discr_perf_array[i] - std_discrimination_array[i], discr_perf_array[i] + std_discrimination_array[i], alpha=0.2)
    
    # Plot Discrimination Performances
    for i, scenario in enumerate(scenarios_meta):
        axes[1].plot(scaling_metric, discr_perf_array_meta[i], label=scenario)
        axes[1].fill_between(scaling_metric, discr_perf_array_meta[i] - std_discrimination_array_meta[i], discr_perf_array_meta[i] + std_discrimination_array_meta[i], alpha=0.2)
    
    ''' 
    # Set the Y-axis range for the second subplot
    discr_first_value = discr_perf_array[0][0]
    discr_stdv = np.std(discr_perf_array)
    axes[1].set_ylim([discr_first_value - 5 * discr_stdv, discr_first_value + 5 * discr_stdv])
    '''
    axes[1].set_ylim([0, 1])

    axes[1].set_title('Discrimination Performances',fontsize=26)
    axes[1].set_xlabel(x_axis,fontsize=24)
    axes[1].set_ylabel('Performance',fontsize=24)
    axes[1].legend(fontsize=16)
    axes[1].set_xscale('log')  # Set x-axis to log scale
    
    axes[0].tick_params(axis='y', labelsize=20)
    axes[0].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)
    axes[1].tick_params(axis='x', labelsize=20)

    plt.tight_layout()
    plt.savefig('NeurIPS_v2_AGL_Task_scaling_{}.png'.format(x_axis.replace(" ", "_").replace("/", "_")), bbox_inches='tight')

    plt.show()
    plt.close(fig)




def plot_violin(list_violin_1st, list_violin_2nd, list_violin_1st_cascade, list_violin_2nd_cascade, plot_title,reference):
  
    reference='Maps'
    # Combine all data into a single list
    print("high consciousness regular, mean:", np.mean(np.array(list_violin_1st)) ,"std",np.std(np.array(list_violin_1st)))
    print("low consciousness regular, mean:", np.mean(np.array(list_violin_2nd)) ,"std",np.std(np.array(list_violin_2nd)) )
    
    print("high consciousness regular, meta:", np.mean(np.array(list_violin_1st_cascade)) ,"std",np.std(np.array(list_violin_1st_cascade)))
    print("low consciousness regular, meta:", np.mean(np.array(list_violin_2nd_cascade)) ,"std",np.std(np.array(list_violin_2nd_cascade)) )
    
    data = [list_violin_1st, list_violin_2nd, list_violin_1st_cascade, list_violin_2nd_cascade]
    
    # Define labels and colors for each violin
    labels = [
        'High Consciousness\n Networks', 
        'Low Consciousness\n Networks', 
        'High Consciousness\n ('+reference+')', 
        'Low Consciousness\n ('+reference+')'
    ]
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=data, palette=colors, ax=ax)

    # Set the x-tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=16)


    # Set the title and labels
    ax.set_title(plot_title, fontsize=26)
    ax.set_ylabel('Metric Values', fontsize=20)
    ax.set_xlabel('Networks', fontsize=20)

    # Calculate Z-scores for each comparison
    def calculate_z_score(data1, data2):
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        z_score = (mean2 - mean1) / pooled_se
        return z_score

    z_high = calculate_z_score(list_violin_1st, list_violin_1st_cascade)
    z_low = calculate_z_score(list_violin_2nd, list_violin_2nd_cascade)

    # Determine significance
    significance_high = abs(z_high) > norm.ppf(0.975)
    significance_low = abs(z_low) > norm.ppf(0.975)

    # Create a table with the results
    table_data = {
        'Condition': ['High Consciousness', 'Low Consciousness'],
        'Z-Score': [z_high, z_low],
        'Significant': [significance_high, significance_low]
    }

    df_results = pd.DataFrame(table_data)

    # Plot the table below the violin plot
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[4, 1])

    ax_violin = fig.add_subplot(gs[0])
    sns.violinplot(data=data, palette=colors, ax=ax_violin)
    ax_violin.set_xticks(range(len(labels)))
    
    
    ax_violin.set_xticklabels(labels, fontsize=18)
    ax_violin.tick_params(axis='y', labelsize=20)
    ax_violin.set_title(plot_title, fontsize=28)
    ax_violin.set_ylabel('Metric Values', fontsize=24)
    ax_violin.set_xlabel('Networks', fontsize=24)

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('tight')
    ax_table.axis('off')
    

    
    table = ax_table.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)  
    table.set_fontsize(20) 
    table.scale(1, 2)
    for key, cell in table.get_celld().items():
        cell.set_fontsize(20) 
        
        
    plt.tight_layout()
    plt.savefig('NeurIPS_v2_AGL_violin_plot_with_table_{}.png'.format(plot_title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_violin_conference(list_violin_1_supra, list_violin_1_sub, list_violin_1_low,
                list_violin_1_cascade_supra, list_violin_1_cascade_sub, list_violin_1_cascade_low, plot_title, type_plot):
    
    print(plot_title)
    print("conf 1, mean:", np.mean(np.array(list_violin_1_supra)), "std:", np.std(np.array(list_violin_1_supra)))
    print("conf 2, mean:", np.mean(np.array(list_violin_1_sub)), "std:", np.std(np.array(list_violin_1_sub)))
    print("conf 3, mean:", np.mean(np.array(list_violin_1_low)), "std:", np.std(np.array(list_violin_1_low)))

    print("conf 4, mean:", np.mean(np.array(list_violin_1_cascade_supra)), "std:", np.std(np.array(list_violin_1_cascade_supra)))
    print("conf 5, mean:", np.mean(np.array(list_violin_1_cascade_sub)), "std:", np.std(np.array(list_violin_1_cascade_sub)))
    print("conf 6, mean:", np.mean(np.array(list_violin_1_cascade_low)), "std:", np.std(np.array(list_violin_1_cascade_low)))

    
    
    # Combine all data into a single list
    data = [list_violin_1_supra, list_violin_1_sub, list_violin_1_low,
            list_violin_1_cascade_supra, list_violin_1_cascade_sub, list_violin_1_cascade_low]

    if type_plot==0:
        # Define labels and colors for each violin
        labels = [
            'Suprathreshold\n Stimuli',
            'Subthreshold\n Stimuli',
            'Low Vision',
            'Suprathreshold\n (Maps)',
            'Subthreshold\n (Maps)',
            'Low Vision\n  (Maps)'
        ]
    elif type_plot==1:
        # Define labels and colors for each violin
        labels = [
            'Setting 1',
            'Setting 2',
            'Setting 3',
            'Setting 4',
            'Setting 5',
            'Setting 6'
        ]
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'lightblue', 'lightpink']

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.violinplot(data=data, palette=colors, ax=ax)

    # Set the x-tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)

    # Set the title and labels
    ax.set_title(plot_title, fontsize=16)
    ax.set_ylabel('Metric Values', fontsize=14)
    ax.set_xlabel('Conditions', fontsize=14)

    # Calculate Z-scores for each comparison
    def calculate_z_score(data1, data2):
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        z_score = (mean2 - mean1) / pooled_se
        return z_score

    if type_plot==0: 
        z_supra = calculate_z_score(list_violin_1_supra, list_violin_1_cascade_supra)
        z_sub = calculate_z_score(list_violin_1_sub, list_violin_1_cascade_sub)
        z_low = calculate_z_score(list_violin_1_low, list_violin_1_cascade_low)

        # Determine significance
        significance_supra = abs(z_supra) > norm.ppf(0.975)
        significance_sub = abs(z_sub) > norm.ppf(0.975)
        significance_low = abs(z_low) > norm.ppf(0.975)

        # Create a table with the results
        table_data = {
            'Condition': ['Suprathreshold', 'Subthreshold', 'Low Vision'],
            'Z-Score': [z_supra, z_sub, z_low],
            'Significant': [significance_supra, significance_sub, significance_low]
        }

        
    elif type_plot==1:
        z_conf1= calculate_z_score(list_violin_1_supra, list_violin_1_sub)
        z_conf2= calculate_z_score(list_violin_1_supra, list_violin_1_low)
        z_conf3= calculate_z_score(list_violin_1_supra, list_violin_1_cascade_supra)
        z_conf4= calculate_z_score(list_violin_1_supra, list_violin_1_cascade_sub)
        z_conf5= calculate_z_score(list_violin_1_supra, list_violin_1_cascade_low)
        
        significance_conf1 = abs(z_conf1) > norm.ppf(0.975)
        significance_conf2 = abs(z_conf2) > norm.ppf(0.975)
        significance_conf3 = abs(z_conf3) > norm.ppf(0.975)
        significance_conf4 = abs(z_conf4) > norm.ppf(0.975)
        significance_conf5 = abs(z_conf5) > norm.ppf(0.975)
        
        table_data = {
            'Condition': ['vs Setting 2', 'vs Setting 3', 'vs Setting 4', 'vs Setting 5', 'vs Setting 6'],
            'Z-Score': [z_conf1, z_conf2, z_conf3, z_conf4, z_conf5],
            'Significant': [significance_conf1, significance_conf2, significance_conf3, significance_conf4, significance_conf5]
        }
        
    
    df_results=pd.DataFrame(table_data)
        

    # Plot the table below the violin plot
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 1, height_ratios=[4, 1])

    ax_violin = fig.add_subplot(gs[0])
    sns.violinplot(data=data, palette=colors, ax=ax_violin)
    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, fontsize=18)
    ax_violin.tick_params(axis='y', labelsize=20)
    ax_violin.set_title(plot_title, fontsize=28)
    ax_violin.set_ylabel('Metric Values', fontsize=24)
    ax_violin.set_xlabel('Conditions', fontsize=24)

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('tight')
    ax_table.axis('off')

    table = ax_table.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)  
    table.set_fontsize(20) 
    table.scale(1, 2)
    for key, cell in table.get_celld().items():
        cell.set_fontsize(20) 
        
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig('IJCAI_violin_plot_AGL_with_table_{}.png'.format(plot_title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
    plt.show()
    plt.close(fig)



def load_data_from_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        data = []
        for row in reader:
            # Convert the string representations of lists back to actual lists
            title = row[0]
            list1 = ast.literal_eval(row[1])
            list2 = ast.literal_eval(row[2])
            list3 = ast.literal_eval(row[3])
            list4 = ast.literal_eval(row[4])
            label = row[5]
            data.append((title, list1, list2, list3, list4, label))
    return data

# Function to load data from the CSV file
def load_violin_data_from_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        data = []
        for row in reader:
            title = row[0]
            list1 = ast.literal_eval(row[1])
            list2 = ast.literal_eval(row[2])
            list3 = ast.literal_eval(row[3])
            cascade_list1 = ast.literal_eval(row[4])
            cascade_list2 = ast.literal_eval(row[5])
            cascade_list3 = ast.literal_eval(row[6])
            data.append((title, list1, list2, list3, cascade_list1, cascade_list2, cascade_list3))
    return data
  
globals()['array'] = np.array


def load_scaling_data_from_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        data = []
        for row in reader:
            scaling_factor = ast.literal_eval(row[0])
            f1_scores_list = ast.literal_eval(row[1])
            discr_perf_list = ast.literal_eval(row[2])
            std_wager_list = ast.literal_eval(row[3])
            std_discrimination_list = ast.literal_eval(row[4])
            f1_scores_list_meta = ast.literal_eval(row[5])
            discr_perf_list_meta = ast.literal_eval(row[6])
            std_wager_list_meta = ast.literal_eval(row[7])
            std_discrimination_list_meta = ast.literal_eval(row[8])
            label = row[9]
            data.append((
                scaling_factor, f1_scores_list, discr_perf_list, std_wager_list, std_discrimination_list,
                f1_scores_list_meta, discr_perf_list_meta, std_wager_list_meta, std_discrimination_list_meta, label
            ))
    return data


def convert_arrays_to_lists(data):
    """Recursively converts NumPy arrays to Python lists."""
    if isinstance(data, list):
        return [convert_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data



def run_setting_experiment(setting, scaling_factors, factors_second, default_hidden_first, default_hidden_second, seeds_violin, cascade_off, cascade_mode):
    """
    Run experiments for a single setting and return the collected data for both high and low consciousness
    """
    # Initialize data structures for high and low consciousness
    high_consciousness_data = [[] for _ in range(len(factors_second))]
    low_consciousness_data = [[] for _ in range(len(factors_second))]
    high_wagering_data = [[] for _ in range(len(factors_second))]
    low_wagering_data = [[] for _ in range(len(factors_second))]
    
    # Standard deviation data
    high_consciousness_std = [[] for _ in range(len(factors_second))]
    low_consciousness_std = [[] for _ in range(len(factors_second))]
    high_wagering_std = [[] for _ in range(len(factors_second))]
    low_wagering_std = [[] for _ in range(len(factors_second))]
    
    # For each primary scaling factor
    for p_idx, factor in enumerate(scaling_factors):
        # For each secondary scaling factor
        for s_idx, factor_second in enumerate(factors_second):
            
            hidden_first_scaling = int(default_hidden_first)
            hidden_second_scaling = int(default_hidden_first*factor_second)
            data_factor = factor

            # Select the appropriate training function based on setting
            if setting == 1:
                # No meta-learning, no cascade
                list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_second=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=1, 
                    gam=0.999, 
                    meta=False, 
                    cascade_rate=cascade_off, 
                    seeds=seeds_violin, 
                    type_cascade=4
                )
            elif setting == 2:
                # No meta-learning, with cascade type 2
                list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_second=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=1, 
                    gam=0.999, 
                    meta=False, 
                    cascade_rate=cascade_mode, 
                    seeds=seeds_violin, 
                    type_cascade=2
                )
            elif setting == 3:
                # With meta-learning, no cascade
                list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_second=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=1, 
                    gam=0.999, 
                    meta=True, 
                    cascade_rate=cascade_off, 
                    seeds=seeds_violin, 
                    type_cascade=4
                )
            elif setting == 4:
                # With meta-learning, with cascade type 2
                list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_second=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=1, 
                    gam=0.999, 
                    meta=True, 
                    cascade_rate=cascade_mode, 
                    seeds=seeds_violin, 
                    type_cascade=2
                )
            elif setting == 5:
                # With meta-learning, with cascade type 3
                list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_second=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=1, 
                    gam=0.999, 
                    meta=True, 
                    cascade_rate=cascade_mode, 
                    seeds=seeds_violin, 
                    type_cascade=3
                )
            elif setting == 6:
                # With meta-learning, with cascade type 1
                list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_second=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=1, 
                    gam=0.999, 
                    meta=True, 
                    cascade_rate=cascade_mode, 
                    seeds=seeds_violin, 
                    type_cascade=1
                )
            
            # Store results: high consciousness [0] and low consciousness [1]
            high_consciousness_data[s_idx].append(list_violin_1st[0])  # Main task performance - high consciousness
            low_consciousness_data[s_idx].append(list_violin_1st[1])   # Main task performance - low consciousness
            high_wagering_data[s_idx].append(list_violin_2nd[0])       # Wagering performance - high consciousness
            low_wagering_data[s_idx].append(list_violin_2nd[1])        # Wagering performance - low consciousness
            
            # Store standard deviations
            high_consciousness_std[s_idx].append(list_1s_metric_std[0])
            low_consciousness_std[s_idx].append(list_1s_metric_std[1])
            high_wagering_std[s_idx].append(list_2s_metric_std[0])
            low_wagering_std[s_idx].append(list_2s_metric_std[1])
            
            # Save individual experiment result
            experiment_log = {
                'primary_factor': factor,
                'secondary_factor': factor_second,
                'high_consciousness_performance': list_violin_1st[0],
                'low_consciousness_performance': list_violin_1st[1],
                'high_wagering_performance': list_violin_2nd[0],
                'low_wagering_performance': list_violin_2nd[1],
                'high_consciousness_std': list_1s_metric_std[0],
                'low_consciousness_std': list_1s_metric_std[1],
                'high_wagering_std': list_2s_metric_std[0],
                'low_wagering_std': list_2s_metric_std[1],
                'meta': setting >= 3,  # settings 3-6 use meta=True
                'cascade_type': [4, 2, 4, 2, 3, 1][setting-1],  # Map setting to cascade type
                'hidden_first': default_hidden_first*factor,
                'hidden_second': default_hidden_second*factor_second
            }
            
            # Save individual experiment details
            log_filename = f'AGL_experiment_factor_{factor}_secondary_{factor_second}_setting_{setting}.csv'
            with open(log_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(experiment_log.keys())  # Header
                writer.writerow([convert_arrays_to_lists(val) if isinstance(val, (list, np.ndarray)) 
                                else val for val in experiment_log.values()])  # Values
            
            print(f"Completed experiment with setting {setting}, primary factor {factor} and secondary factor {factor_second}")
    
    # Save results for this setting
    setting_data = {
        'scaling_factors': scaling_factors,
        'factors_second': factors_second,
        'high_consciousness_data': convert_arrays_to_lists(high_consciousness_data),
        'low_consciousness_data': convert_arrays_to_lists(low_consciousness_data),
        'high_wagering_data': convert_arrays_to_lists(high_wagering_data),
        'low_wagering_data': convert_arrays_to_lists(low_wagering_data),
        'high_consciousness_std': convert_arrays_to_lists(high_consciousness_std),
        'low_consciousness_std': convert_arrays_to_lists(low_consciousness_std),
        'high_wagering_std': convert_arrays_to_lists(high_wagering_std),
        'low_wagering_std': convert_arrays_to_lists(low_wagering_std),
        'setting': setting
    }
    
    # Save setting results to CSV
    log_filename = f'AGL_scaling_plot_data_setting_{setting}.csv'
    with open(log_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(setting_data.keys())  # Header
        writer.writerow([convert_arrays_to_lists(val) if isinstance(val, (list, np.ndarray)) 
                        else val for val in setting_data.values()])  # Values
    
    print(f"Setting {setting} data saved to '{log_filename}'.")
    
    return high_consciousness_data, low_consciousness_data, high_wagering_data, low_wagering_data, high_consciousness_std, low_consciousness_std, high_wagering_std, low_wagering_std


def plot_scaling_consciousness(scaling_factors, all_data, factors_second, settings, consciousness_level, data_type="task", std_data=None):
    """
    Create a plot with 6 subplots showing performance across settings for specified consciousness level
    
    Args:
        scaling_factors: List of primary scaling factors (x-axis)
        all_data: Dictionary where keys are settings (1-6) and values are nested lists of performance values
        factors_second: List of secondary factor values (for legend)
        settings: List of settings to plot (1-6)
        consciousness_level: String indicating "high" or "low" consciousness level
        data_type: String indicating "task" or "wagering" performance
        std_data: Dictionary with same structure as all_data, containing standard deviation values
    """

    # Create figure with 6 subplots (2 columns, 3 rows)
    fig, axs = plt.subplots(3, 2, figsize=(16, 18), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(factors_second)))
    
    # Determine title components based on params
    consciousness_text = "High Consciousness" if consciousness_level.lower() == "high" else "Low Consciousness"
    performance_text = "Task Performance" if data_type.lower() == "task" else "Wagering Performance"
    
    # For each setting (1-6)
    for setting_idx, setting in enumerate(settings):
        ax = axs[setting_idx]
        
        # Get data for this setting
        setting_data = all_data.get(setting, [[] for _ in range(len(factors_second))])
        
        # Get standard deviation data for this setting if available
        setting_std_data = None
        if std_data is not None:
            setting_std_data = std_data.get(setting, [[] for _ in range(len(factors_second))])
        
        # Plot each secondary factor line
        for i, factor_second in enumerate(factors_second):
            # For data with nested lists, compute means first
            if setting_data[i] and isinstance(setting_data[i][0], list):
                # Calculate means for each scaling factor
                mean_data = []
                for scale_idx in range(len(scaling_factors)):
                    # For each scaling factor, get all trial values
                    trials = []
                    if scale_idx < len(setting_data[i]):
                        trials = setting_data[i][scale_idx]
                    # Calculate mean for this scaling factor
                    if trials:
                        mean_data.append(sum(trials) / len(trials))
                    else:
                        mean_data.append(0)  # Handle empty data
                plot_data = mean_data
            else:
                # Data already in correct format
                plot_data = setting_data[i]
            
            #full  list  [0.1, 0.2 , 0.5, 1.0, 2.0, 5.0]
            if factor_second in  [0.5, 1.0, 5.0]:
              # Plot the main line
              ax.plot(
                  scaling_factors, 
                  plot_data,
                  marker='o',
                  linestyle='-',
                  color=colors[i],
                  label=f'Factor = {factor_second}'
              )
              
              # Add standard deviation bands if available
              if setting_std_data is not None and len(setting_std_data[i]) > 0:
                  # Check if std data is a list or a single value
                  if isinstance(setting_std_data[i], (int, float)):
                      # Use the same std for all points
                      std_values = [setting_std_data[i]] * len(plot_data)
                  else:
                      # Use the provided std values
                      std_values = setting_std_data[i]


                  # Calculate upper and lower bounds
                  upper_bound = [d + s for d, s in zip(plot_data, std_values)]
                  lower_bound = [d - s for d, s in zip(plot_data, std_values)]
                  
                  # Plot the filled area between upper and lower bounds
                  ax.fill_between(
                      scaling_factors,
                      lower_bound,
                      upper_bound,
                      color=colors[i],
                      alpha=0.2  # Transparency
                  )
        
        # Use log scale for x-axis if there are multiple scaling factors with wide range
        if len(scaling_factors) > 2 and max(scaling_factors) / min(scaling_factors) > 10:
            ax.set_xscale('log', base=2)
        
        ax.set_title(f'Setting {setting}')
        ax.grid(True, alpha=0.3)
        
        # Add setting-specific descriptions
        setting_descriptions = {
            1: "No 2nd-net, baseline",
            2: "No 2nd-net, cascade",
            3: "2nd-net, no cascade",
            4: "2nd-net, cascade 1st",
            5: "2nd-net, cascade 2nd",
            6: "2nd-net, cascade both"
        }
        
        ax.text(0.05, 0.95, setting_descriptions.get(setting, ""), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add common labels for all subplots
    fig.text(0.5, 0.08, 'Primary Scaling Factor', ha='center', fontsize=14)
    fig.text(0.08, 0.5, f'{performance_text} ({consciousness_text})', va='center', rotation='vertical', fontsize=14)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(factors_second), fontsize=12, frameon=True)
    
    plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.92])  # Adjust layout to make room for common labels
    
    # Add overall title
    plt.suptitle(f'AGL {performance_text} ({consciousness_text}) by Scaling Factor Across Settings', fontsize=16, y=0.995)
    
    # Create filename based on parameters
    consciousness_filename = "high" if consciousness_level.lower() == "high" else "low"
    performance_filename = "task" if data_type.lower() == "task" else "wagering"
    
    # Save the plot
    filename = f'AGL_{consciousness_filename}_{performance_filename}_performance.png'
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"{consciousness_text} {performance_text} plot saved as '{filename}'.")
    
def load_setting_data_from_csv(setting):
    """
    Load experiment data for a specific setting from CSV file
    Returns a dictionary with the loaded data
    """
    import csv
    import os
    import ast
    import re
    
    # Increase the CSV field size limit
    csv.field_size_limit(2147483647)  # Set to maximum possible value
    
    filename = f'AGL_scaling_plot_data_setting_{setting}.csv'
    
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found")
        return None
    
    try:
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            keys = next(reader)  # Read header row
            values = next(reader)  # Read data row
            
            # Convert string representations back to Python objects
            data = {}
            
            for i, key in enumerate(keys):
                value_str = values[i]
                
                try:
                    # For simple lists
                  data[key] = ast.literal_eval(value_str)
                
                except (SyntaxError, ValueError) as e:
                    # For numpy arrays with float64 values
                    try:
                        # Convert the numpy-specific format to actual numerical values
                        # Extract numbers from np.float64(...) patterns
                        float_pattern = r'np\.float64\(([^)]+)\)'
                        matches = re.findall(float_pattern, value_str)
                        
                        if matches:
                            # Determine the structure based on bracket levels
                            # Count opening brackets to determine nesting level
                            nesting_level = value_str.count('[[[')
                            
                            if nesting_level >= 1:
                                # Handle 3D array [[[...]]]
                                # Split by closing/opening brackets to get the different sub-arrays
                                sublists = re.findall(r'\[\[(.*?)\]\]', value_str)
                                result = []
                                for sublist in sublists:
                                    # Extract all float values
                                    floats = re.findall(float_pattern, sublist)
                                    if floats:
                                        result.append([float(x) for x in floats])
                                data[key] = [result]  # Wrap in list to maintain 3D structure
                            else:
                                # Handle 2D array [[...]]
                                floats = [float(x) for x in matches]
                                data[key] = [floats]
                        else:
                            print(f"No float values found for key '{key}'")
                            data[key] = []
                            
                    except Exception as e:
                        print(f"Error parsing data for key '{key}': {e}")
                        # As a fallback, store the raw string
                        data[key] = value_str
        
        return data
        
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

def main():
    """
    Main function to run experiments on artificial grammar learning (AGL) with metacognitive networks.
    
    This implementation uses:
    - First-order network (1st-Net): An auto-encoder for pattern recognition tasks
    - Second-order network (2nd-Net): A comparator matrix connected to two wagering units
    - Contrastive loss for the main task to provide information flow for wagering
    - Cross-entropy loss for wagering to handle class imbalance
    - Cascade model in both networks for smooth graded accumulation of activation
    
    For AGL tasks:
    - Pre-training: Uses random strings to establish initial weights
    - Training: Uses grammar A strings
    - Testing: Uses a mix of grammar A and grammar B strings
    - Low training scheme with just 3 epochs
    
    The results show statistical significance with Z-scores of 15.0 for MAPS and 4.2 for 2nd-Net.
    """
    # Define hyperparameter search space
    hidden_sizes = [40, 50, 80, 100, 150, 160]  # Size of hidden layers
    factors = [1]  # Data scale factor (multiplier for dataset size)
    gelus = [True, False]  # Activation function: True for GELU, False for ReLU
    step_sizes = [1, 2]  # Learning rate scheduler step size (epochs)
    gammas = [0.99, 0.98, 0.97]  # Learning rate decay factor
    metalayers = [True]  # Whether to enable training of the second-order (metacognitive) layer
    num_iterations = 1  # Number of training runs per configuration

    # Seeds for experiments
    seeds_violin = 45  # For violin plot experiments (45*10 (number of different trained networks per iteration) = 450 total runs) 

    # Cascade parameters
    cascade_mode = 0.02  # Cascade rate when enabled (smaller values mean more iterations)
    cascade_off = 1.0    # Value to effectively disable cascade (single iteration)

    # Tracking variables for hyperparameter optimization
    best_mean_value = 0
    best_max_high_value = 0
    best_max_low_value = 0
    hyperparameters = list(product(hidden_sizes, factors, gelus, step_sizes, gammas, metalayers))
    
    # Experiment control flags
    Training = False
  
    # Cascade types explanation:
    # cascade type 1: Both 1st and 2nd order networks use cascade mode
    # cascade type 2: Only 1st order network uses cascade model
    # cascade type 3: Only 2nd order network uses cascade model
    # cascade type 4: Neither network uses cascade model (baseline)
    
    if Training:
        print("starting training")
        
        # Run 1: Baseline - No metacognition, no cascade model (type 4)
        list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data2, list_violin_1st_no_meta, list_violin_2nd_no_meta = train(
            40, 48, 1, False, 1, 0.999, False, cascade_off, seeds_violin, 4)

        # Run 2: No metacognition, with cascade model on 1st-Net only (type 2)
        list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data2, list_violin_1st_no_meta_conf2, list_violin_2nd_no_meta_conf2 = train(
            40,48, 1, False, 1, 0.999, False, cascade_mode, seeds_violin, 2)

        # Run 3: With metacognition, no cascade model (type 4)
        list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data3, list_violin_1st_cascade_conf3, list_violin_2nd_cascade_conf3 = train(
            40, 48,1, False, 1, 0.999, True, cascade_off, seeds_violin, 4)

        # Run 4: With metacognition, cascade model on 1st-Net only (type 2)
        list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data3, list_violin_1st_cascade_conf4, list_violin_2nd_cascade_conf4 = train(
            40, 48,1, False, 1, 0.999, True, cascade_mode, seeds_violin, 2)

        # Run 5: With metacognition, cascade model on 2nd-Net only (type 3)
        list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data3, list_violin_1st_cascade_conf5, list_violin_2nd_cascade_conf5 = train(
            40, 48,  1, False, 1, 0.999, True, cascade_mode, seeds_violin, 3)

        # Run 6: With metacognition, cascade model on both networks (type 1) - full model
        list_1s_metric, list_2s_metric, list_1s_metric_std, list_2s_metric_std, max_value_high, max_value_low, mean_value, random_baseline_testing, networks, plot_data3, list_violin_1st_cascade, list_violin_2nd_cascade = train(
            40,48, 1, False, 1, 0.999, True, cascade_mode, seeds_violin, 1)

        # Prepare data for CSV logging and violin plots
        # Each row contains results for a different metric across all experimental conditions
        log_data = [
            # High consciousness results for main task (discrimination performance)
            ['Discrimination Performance (main task) - High consciousness', 
             list_violin_1st_no_meta[0], list_violin_1st_no_meta_conf2[0], 
             list_violin_1st_cascade_conf3[0], list_violin_1st_cascade_conf4[0], 
             list_violin_1st_cascade_conf5[0], list_violin_1st_cascade[0]],
            
            # High consciousness results for metacognitive task (wagering accuracy)
            ['Accuracy (wagering) - High consciousness', 
             list_violin_2nd_no_meta[0], list_violin_2nd_no_meta_conf2[0], 
             list_violin_2nd_cascade_conf3[0], list_violin_2nd_cascade_conf4[0], 
             list_violin_2nd_cascade_conf5[0], list_violin_2nd_cascade[0]],
            
            # Low consciousness results for main task (discrimination performance)
            ['Discrimination Performance (main task) - low consciousness', 
             list_violin_1st_no_meta[1], list_violin_1st_no_meta_conf2[1], 
             list_violin_1st_cascade_conf3[1], list_violin_1st_cascade_conf4[1], 
             list_violin_1st_cascade_conf5[1], list_violin_1st_cascade[1]],
            
            # Low consciousness results for metacognitive task (wagering accuracy)
            ['Accuracy (wagering) - low consciousness', 
             list_violin_2nd_no_meta[1], list_violin_2nd_no_meta_conf2[1], 
             list_violin_2nd_cascade_conf3[1], list_violin_2nd_cascade_conf4[1], 
             list_violin_2nd_cascade_conf5[1], list_violin_2nd_cascade[1]]
        ]
        
        # Save results to CSV file for later analysis
        with open('AGL_violin_plot_logs_conference.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write CSV header with column names
            writer.writerow(['Title', 'List 1', 'List 2', 'List 3', 'List 4', 'List 5', 'List 6'])
            
            # Write the experimental data
            writer.writerows(log_data)

        print("Variables saved to 'AGL_violin_plot_logs_conference.csv'.")
      
        # Load the saved experimental data
        violin_data = load_violin_data_from_csv('AGL_violin_plot_logs_conference.csv')

        # Generate violin plots for each metric across all experimental conditions
        for data in violin_data:
            title, list1, list2, list3, list4, list5, list6 = data
            plot_violin_conference(list1, list2, list3, list4, list5, list6, title, 1)


    # Configuration
    default_hidden_first = 40
    default_hidden_second = 48
    scaling_factors = [1, 2, 3, 5, 10,  15, 25, 50, 100]
    factors_second = [0.1, 0.2 , 0.5, 1.0, 2.0, 5.0]
    seeds_scaling = 10  # Number of seeds for scaling experiments
    
    scaling=True
    
    load_data=True
    
    labels=['high_consciousness_data', 'low_consciousness_data', 'high_wagering_data','low_wagering_data','high_consciousness_std','low_consciousness_std','high_wagering_std','low_wagering_std']
    
    
    all_data = {label: {} for label in labels}

    
    if scaling:
      # Run experiments for all 6 settings
      for setting in range(1, 7):
        
          if load_data==True:
            
            print(f"Loading data for setting {setting}")
            # Load data from CSV
            setting_data = load_setting_data_from_csv(setting)
                       # Store data in respective dictionaries

            for label in labels:
                # Initialize the setting key with an empty list
                all_data[label][setting] = []
                
                if len(setting_data[label]) == 1:

                    flat_data = setting_data[label][0]
                      
                    # Reorganize the data
                    for i, factor_second in enumerate(factors_second):
                        all_data[label][setting].append([])  # Create a list for this factor_second
                        
                        factor_data = flat_data[i]  # Get the data for this factor_second
                        
                        # Split the 900 values into 9 sublists of 100 values each
                        for j, scaling_factor in enumerate(scaling_factors):
                            #if 'std' not in label:
                            #  all_data[label][setting][i].append([])  # Create a list for this factor_second

                            if 'std' not in label:
                              start_idx = j * seeds_scaling
                              end_idx = start_idx + seeds_scaling
                              all_data[label][setting][i].append(factor_data[start_idx:end_idx])

                            #if 'std' not in label:
                            #  for k in range(int(len(factor_data)/len(scaling_factors))):
                            #    all_data[label][setting][i][j].append(factor_data[k*len(scaling_factors) + j ])
                            #else:
                            else:
                              all_data[label][setting][i].append(flat_data[i*len(scaling_factors)+j]) 
                      
                else:
                    all_data[label][setting] = setting_data[label]

            
                        
          else:
            
            print(f"Starting experiments for setting {setting}")
            
            high_consciousness, low_consciousness, high_wagering, low_wagering, \
            high_consciousness_std, low_consciousness_std, high_wagering_std, low_wagering_std = run_setting_experiment(
                setting, scaling_factors, factors_second, default_hidden_first, 
                default_hidden_second, seeds_scaling, cascade_off, cascade_mode
            )
          
            # Store results
            all_data[labels[0]][setting]= high_consciousness
            all_data[labels[1]][setting]= low_consciousness
            all_data[labels[2]][setting]= high_wagering
            all_data[labels[3]][setting]= low_wagering
            all_data[labels[4]][setting]= high_consciousness_std
            all_data[labels[5]][setting]= low_consciousness_std
            all_data[labels[6]][setting]= high_wagering_std
            all_data[labels[7]][setting]= low_wagering_std
            
      
      # Create plots for high consciousness task performance
      plot_scaling_consciousness(
          scaling_factors,
          all_data['high_consciousness_data'],
          factors_second,
          list(range(1, 7)),  # Settings 1-6
          consciousness_level="high",
          data_type="task",
          std_data=all_data['high_consciousness_std']
      )
      
      # Create plots for low consciousness task performance
      plot_scaling_consciousness(
          scaling_factors,
          all_data['low_consciousness_data'],
          factors_second,
          list(range(1, 7)),  # Settings 1-6
          consciousness_level="low",
          data_type="task",
          std_data=all_data['low_consciousness_std']
      )
    
# Execute the main function
main()