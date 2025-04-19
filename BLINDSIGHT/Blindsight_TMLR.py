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
import math
import csv
import ast  # To safely evaluate strings as Python expressions
import re
import torch_optimizer as optim2
import os
from torchmetrics.functional.regression import mean_absolute_percentage_error
from torch.autograd import Variable

from scipy.optimize import curve_fit

from scipy.stats import norm

import torch_optimizer as optim2


# %% [markdown]
# # DEVICE CONFIGURATION

# %%
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

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

        self.fc1 = nn.Linear(100, hidden_units, bias = False) # Encoder
        self.fc2 = nn.Linear(hidden_units, 100, bias = False) # Decoder

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)

        # Set the data factor
        self.data_factor = data_factor

        # Other activation functions for various purposes
        self.softmax = nn.Softmax()

        # Initialize network weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights of the encoder, hidden, and decoder layers uniformly."""
        init.uniform_(self.fc1.weight, -1.0, 1.0)
        init.uniform_(self.fc2.weight, -1.0, 1.0)

    def encoder(self, x):
      h1 = self.dropout(self.relu(self.fc1(x.view(-1, 100))))
      return h1

    def decoder(self,z, prev_h2 , cascade_rate):
      h2 = self.sigmoid(self.fc2(z))

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
    def __init__(self, use_gelu, hidden_2nd):
        super(SecondOrderNetwork, self).__init__()
        # Define a linear layer for comparing the difference between input and output of the first-order network
        self.comparison_layer = nn.Linear(100, hidden_2nd)

        # Linear layer for determining wagers, mapping from 100 features to a single output
        self.wager = nn.Linear(hidden_2nd, 1)

        # Dropout layer to prevent overfitting by randomly setting input units to 0 with a probability of 0.5 during training
        self.dropout = nn.Dropout(0.5)

        # Select activation function based on the `use_gelu` flag
        self.activation = torch.relu

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
# #FUNCTION THAT CREATES PATTERS AND TARGETS FOR EACH EPOCH
# 

# %%
def generate_patterns(patterns_number, num_units, factor, condition, noise_level):
    """
    Generates patterns and targets for training the networks

    # patterns_number: Number of patterns to generate
    # num_units: Number of units in each pattern
    # pattern: 0: superthreshold, 1: subthreshold, 2: low vision
    # Returns lists of patterns, stimulus present/absent indicators, and second order targets
    """

    patterns_number= patterns_number*factor

    patterns = []  # Store generated patterns
    stim_present = []  # Indicators for when a stimulus is present in the pattern
    stim_absent = []  # Indicators for when no stimulus is present
    order_2_pr = []  # Second order network targets based on the presence or absence of stimulus

    if condition == 0:
        random_limit= 0.0
        baseline = 0
        multiplier = 1

    if condition == 1:
        random_limit= 0.02
        baseline = noise_level
        multiplier = 1

    if condition == 2:
        random_limit= 0.02
        baseline = noise_level
        multiplier = 0.3

    # Generate patterns, half noise and half potential stimuli
    for i in range(patterns_number):

        # First half: Noise patterns
        if i < patterns_number // 2:

            pattern = multiplier * np.random.uniform(0.0, random_limit, num_units) + baseline # Generate a noise pattern
            patterns.append(pattern)
            stim_present.append(np.zeros(num_units))  # Stimulus absent
            order_2_pr.append([0.0 , 1.0])  # No stimulus, low wager

        # Second half: Stimulus patterns
        else:
            stimulus_number = random.randint(0, num_units - 1) # Choose a unit for potential stimulus
            pattern = np.random.uniform(0.0, random_limit, num_units) + baseline
            pattern[stimulus_number] = np.random.uniform(0.0, 1.0) * multiplier   # Set stimulus intensity

            patterns.append(pattern)
            present = np.zeros(num_units)
            # Determine if stimulus is above discrimination threshold
            if pattern[stimulus_number] >= multiplier/2:
                order_2_pr.append([1.0 , 0.0])  # Stimulus detected, high wager
                present[stimulus_number] = 1.0
            else:
                order_2_pr.append([0.0 , 1.0])  # Stimulus not detected, low wager
                present[stimulus_number] = 0.0

            stim_present.append(present)


    patterns_tensor = torch.Tensor(patterns).to(device).requires_grad_(True)
    stim_present_tensor = torch.Tensor(stim_present).to(device).requires_grad_(True)
    stim_absent_tensor= torch.Tensor(stim_absent).to(device).requires_grad_(True)
    order_2_tensor = torch.Tensor(order_2_pr).to(device).requires_grad_(True)

    return patterns_tensor, stim_present_tensor, stim_absent_tensor, order_2_tensor




def plot_signal_max_and_indicator(patterns_tensor, plot_title="Training Signals"):
    """
    Plots the maximum values of signal units and a binary indicator for max values greater than 0.5.

    Parameters:
    - patterns_tensor: A tensor containing signals, where each signal is expected to have multiple units.
    """

    # Calculate the maximum value of units for each signal within the patterns tensor
    max_values_of_units = patterns_tensor.max(dim=1).values.cpu().numpy()  # Ensure it's on CPU and in NumPy format for plotting

    # Determine the binary indicators based on the max value being greater than 0.5
    binary_indicators = (max_values_of_units > 0.5).astype(int)

    # Create a figure with 2 subplots (2 rows, 1 column)
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    fig.suptitle(plot_title, fontsize=16)  # Set the overall title for the plot

    # First subplot for the maximum values of each signal
    axs[0].plot(range(patterns_tensor.size(0)), max_values_of_units, drawstyle='steps-mid')
    axs[0].set_xlabel('Signal Number')
    axs[0].set_ylabel('Max Value of Signal Units')
    axs[0].grid(True)

    # Second subplot for the binary indicators
    axs[1].plot(range(patterns_tensor.size(0)), binary_indicators, drawstyle='steps-mid', color='red')
    axs[1].set_xlabel('Signal Number')
    axs[1].set_ylabel('Indicator (Max > 0.5) in each signal')
    axs[1].set_ylim(-0.1, 1.1)  # Adjust y-axis limits for clarity
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def create_patterns(stimulus,factor, noise_level):
    """
    Generates neural network input patterns based on specified stimulus conditions.

    Parameters:
    - stimulus (int): Determines the type of patterns to generate.
                      Acceptable values:
                      - 0: Suprathreshold stimulus
                      - 1: Subthreshold stimulus
                      - 2: Low vision condition

    Returns:
    - torch.Tensor: Tensor of generated patterns.
    - torch.Tensor: Tensor of target values corresponding to the generated patterns.
    """

    # Generate initial patterns and target tensors for base condition.

    patterns_tensor, stim_present_tensor, _, _ = generate_patterns(patterns_number, num_units ,factor, stimulus , noise_level)
    # Convert pattern tensors for processing on specified device (CPU/GPU).
    patterns = torch.Tensor(patterns_tensor).to(device)
    targets = torch.Tensor(stim_present_tensor).to(device)

    return patterns, targets



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
# #ASSIGNMENT OF FIRST AND SECOND ORDER NETWORK, AND DEFINITION OF CRITERIONS
# 

# %%
# define the architecture, optimizers, loss functions, and schedulers for pre training
def prepare_pre_training(hidden, hidden_2nd,factor,gelu,stepsize, gam, optimizer):

  first_order_network = FirstOrderNetwork(hidden, factor, gelu).to(device)
  second_order_network = SecondOrderNetwork(gelu, hidden_2nd).to(device)

  criterion_1 = CAE_loss
  criterion_2 = nn.BCELoss(size_average = False)


  if optimizer == "ADAM":
    optimizer_1 = optim.Adam(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Adam(second_order_network.parameters(), lr=learning_rate_2)

  elif optimizer == "SGD":
    optimizer_1 = optim.SGD(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.SGD(second_order_network.parameters(), lr=learning_rate_2)

  elif optimizer == "SWATS":
    optimizer_1 = optim2.SWATS(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim2.SWATS(second_order_network.parameters(), lr=learning_rate_2)

  elif optimizer == "ADAMW":
    optimizer_1 = optim.AdamW(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.AdamW(second_order_network.parameters(), lr=learning_rate_2)

  elif optimizer == "RMS":
    optimizer_1 = optim.RMSprop(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.RMSprop(second_order_network.parameters(), lr=learning_rate_2)

  elif optimizer == "ADAMAX":
    optimizer_1 = optim.Adamax(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim.Adamax(second_order_network.parameters(), lr=learning_rate_2)
    
  elif optimizer== "RANGERVA":
    optimizer_1 = optim2.RangerVA(first_order_network.parameters(), lr=learning_rate_1)
    optimizer_2 = optim2.RangerVA(second_order_network.parameters(), lr=learning_rate_2)


  # Learning rate schedulers
  scheduler_1 = StepLR(optimizer_1, step_size=stepsize, gamma=gam)
  scheduler_2 = StepLR(optimizer_2, step_size=stepsize, gamma=gam)

  return first_order_network, second_order_network, criterion_1 , criterion_2, optimizer_1, optimizer_2, scheduler_1, scheduler_2



# %% [markdown]
# # PRE TRAINING LOOP

# %%
def pre_train(first_order_network, second_order_network, criterion_1,  criterion_2, optimizer_1, optimizer_2, scheduler_1, scheduler_2, factor, meta , noise_level, cascade_rate, type_cascade):
    """
    Conducts pre-training for first-order and second-order networks.

    Parameters:
    - first_order_network (torch.nn.Module): Network for basic input-output mapping.
    - second_order_network (torch.nn.Module): Network for decision-making based on the first network's output.
    - criterion_1, criterion_2 (torch.nn): Loss functions for the respective networks.
    - optimizer_1, optimizer_2 (torch.optim): Optimizers for the respective networks.
    - scheduler_1, scheduler_2 (torch.optim.lr_scheduler): Schedulers for learning rate adjustment.
    - factor (float): Parameter influencing data augmentation or pattern generation.
    - meta (bool): Flag indicating the use of meta-learning strategies.

    Returns:
    Tuple containing updated networks and epoch-wise loss records.


    """
    def get_num_args(func):
      return func.__code__.co_argcount


    max_values_output_first_order = []
    max_indices_output_first_order = []
    max_values_patterns_tensor = []
    max_indices_patterns_tensor = []

    epoch_1_order = np.zeros(n_epochs)
    epoch_2_order = np.zeros(n_epochs)


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

    for epoch in range(n_epochs):


        # Generate training patterns and targets for each epoch
        patterns_tensor, stim_present_tensor, stim_absent_tensor, order_2_tensor = generate_patterns(patterns_number, num_units,factor, 0 , noise_level)

        hidden_representation=None
        output_first_order=None
        comparison_out=None

        for i in range(cascade_iterations_one):

          # Forward pass through the first-order network
          hidden_representation , output_first_order = first_order_network(patterns_tensor, hidden_representation, output_first_order, cascade_rate_one)

        patterns_tensor=patterns_tensor.requires_grad_(True)
        output_first_order=output_first_order.requires_grad_(True)


        ############################################

        # Get max values and indices for output_first_order
        max_vals_out, max_inds_out = torch.max(output_first_order[100:], dim=1)
        max_inds_out[max_vals_out == 0] = 0
        max_values_output_first_order.append(max_vals_out.tolist())
        max_indices_output_first_order.append(max_inds_out.tolist())

        # Get max values and indices for patterns_tensor
        max_vals_pat, max_inds_pat = torch.max(patterns_tensor[100:], dim=1)
        max_inds_pat[max_vals_pat == 0] = 0
        max_values_patterns_tensor.append(max_vals_pat.tolist())
        max_indices_patterns_tensor.append(max_inds_pat.tolist())

        ############################################


        optimizer_1.zero_grad()

        # Conditionally execute the second-order network pass and related operations
        if meta:

            for i in range(cascade_iterations_two):

              # Forward pass through the second-order network with inputs from the first-order network
              output_second_order , comparison_out = second_order_network(patterns_tensor, output_first_order, comparison_out, cascade_rate_two)

            # Calculate the loss for the second-order network (wagering decision based on comparison)
            loss_2 = criterion_2(output_second_order.squeeze(), order_2_tensor[:, 0])

            optimizer_2.zero_grad()


            # Backpropagate the second-order network's loss
            loss_2.backward(retain_graph=True)  # Allows further backpropagation for loss_1 after loss_2

            # Update second-order network weights
            optimizer_2.step()

            scheduler_2.step()

            epoch_2_order[epoch] = loss_2.item()
        else:
            # Skip computations for the second-order network
            with torch.no_grad():
                # Potentially forward pass through the second-order network without tracking gradients
                for i in range(cascade_iterations_two):
                  # Forward pass through the second-order network with inputs from the first-order network
                  output_second_order , comparison_out = second_order_network(patterns_tensor, output_first_order, comparison_out, cascade_rate_two)

        # Calculate the loss for the first-order network (accuracy of stimulus representation)

        num_args = get_num_args(criterion_1)

        if num_args == 2:
          loss_1 = criterion_1(  output_first_order , stim_present_tensor )
        else:
          W = first_order_network.state_dict()['fc1.weight']
          loss_1 = criterion_1( W, stim_present_tensor.view(-1, 100), output_first_order,
                             hidden_representation, lam )


        # Backpropagate the first-order network's loss
        loss_1.backward(retain_graph=True)


        # Update first-order network weights
        optimizer_1.step()

        # Reset first-order optimizer gradients to zero for the next iteration

        # Update the first-order scheduler
        scheduler_1.step()

        epoch_1_order[epoch] = loss_1.item()
        #epoch_1_order[epoch] = loss_location.item()



    return first_order_network, second_order_network, epoch_1_order, epoch_2_order , (max_values_output_first_order[-1],
            max_indices_output_first_order[-1],
            max_values_patterns_tensor[-1],
            max_indices_patterns_tensor[-1])

# %% [markdown]
# # LOSS PLOT

# %%
def perform_quadratic_regression(epoch_list, values):
    # Perform quadratic regression
    coeffs = np.polyfit(epoch_list, values, 2)  # Coefficients of the polynomial
    y_pred = np.polyval(coeffs, epoch_list)        # Evaluate the polynomial at the given x values
    return y_pred

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def perform_exponential_decay_fitting(x_data, y_data):
    # Convert x_data to a numpy array
    x_data = np.array(x_data)
    # Fit the data to the exponential decay model
    popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(1, 1e-6, 1), maxfev=5000)
    y_pred = exponential_decay(x_data, *popt)
    return y_pred

def pre_train_plots(epoch_1_order, epoch_2_order, max_values_indices, title):
    """
    Plots the training progress with regression lines and scatter plots of indices and values of max elements.

    Parameters:
    - epoch_list (list): List of epoch numbers.
    - epoch_1_order (list): Loss values for the first-order network over epochs.
    - epoch_2_order (list): Loss values for the second-order network over epochs.
    - title (str): Title for the plots.
    - max_values_indices (tuple): Tuple containing lists of max values and indices for both tensors.
    """
    (max_values_output_first_order,
     max_indices_output_first_order,
     max_values_patterns_tensor,
     max_indices_patterns_tensor) = max_values_indices

    # Perform exponential decay fitting for the loss plots
    epoch_list = list(range(len(epoch_1_order)))
    y_pred1 = perform_exponential_decay_fitting(epoch_list, epoch_1_order)
    y_pred2 = perform_exponential_decay_fitting(epoch_list, epoch_2_order)

    # Set up the plot with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))  # Set figsize to (10, 10) for a square figure

    # First graph for 1st Order Network
    axs[0, 0].plot(epoch_list, epoch_1_order, linestyle='--', marker='o', color='g')
    axs[0, 0].plot(epoch_list, y_pred1, linestyle='-', color='r', label='Exponential Decay Fit')
    axs[0, 0].legend(['1st Order Network', 'Exponential Decay Fit'])
    axs[0, 0].set_title('1st Order Network Loss')
    axs[0, 0].set_xlabel('Epochs - Pretraining Phase')
    axs[0, 0].set_ylabel('Loss')

    # Second graph for 2nd Order Network
    axs[0, 1].plot(epoch_list, epoch_2_order, linestyle='--', marker='o', color='b')
    axs[0, 1].plot(epoch_list, y_pred2, linestyle='-', color='r', label='Exponential Decay Fit')
    axs[0, 1].legend(['2nd Order Network', 'Exponential Decay Fit'])
    axs[0, 1].set_title('2nd Order Network Loss')
    axs[0, 1].set_xlabel('Epochs - Pretraining Phase')
    axs[0, 1].set_ylabel('Loss')

    # Scatter plot of indices: patterns_tensor vs. output_first_order
    axs[1, 0].scatter(max_indices_patterns_tensor, max_indices_output_first_order, alpha=0.5)

    axs[1, 0].set_title('Stimuli location: First Order Input vs. First Order Output')
    axs[1, 0].set_xlabel('First Order Input Indices')
    axs[1, 0].set_ylabel('First Order Output Indices')
    axs[1, 0].legend()

    # Scatter plot of values: patterns_tensor vs. output_first_order
    axs[1, 1].scatter(max_values_patterns_tensor, max_values_output_first_order, alpha=0.5)

    axs[1, 1].set_title('Stimuli Values: First Order Input vs. First Order Output')
    axs[1, 1].set_xlabel('First Order Input Values')
    axs[1, 1].set_ylabel('First Order Output Values')
    axs[1, 1].legend()

    plt.suptitle(title, fontsize=16, y=1.02)

    # Display the plots in a 2x2 grid
    plt.tight_layout()
    plt.savefig('Blindsight_Pre_training_Loss_{}.png'.format(title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %% [markdown]
# # MODEL LOADING FOR TRAINING

# %%
# Function to configure the training environment and load the models
def config_training(first_order_network, second_order_network, hidden, hidden_2nd, factor, gelu, noise_level):
    """
    Configures the training environment by saving the state of the given models and loading them back.
    Initializes testing patterns for evaluation.

    Parameters:
    - first_order_network: The first order network instance.
    - second_order_network: The second order network instance.
    - hidden: Number of hidden units in the first order network.
    - factor: Factor influencing the network's architecture.
    - gelu: Activation function to be used in the network.

    Returns:
    - Tuple of testing patterns, number of samples in the testing patterns, and the loaded model instances.
    """
    # Paths where the models' states will be saved
    PATH = './cnn1.pth'
    PATH_2 = './cnn2.pth'

    # Save the weights of the pretrained networks to the specified paths
    torch.save(first_order_network.state_dict(), PATH)
    torch.save(second_order_network.state_dict(), PATH_2)

    # Generating testing patterns for three different sets
    First_set, First_set_targets = create_patterns(0,factor , noise_level)
    Second_set, Second_set_targets = create_patterns(1,factor, noise_level)
    Third_set, Third_set_targets = create_patterns(2,factor , noise_level)

    # Aggregate testing patterns and their targets for ease of access
    Testing_patterns = [[First_set, First_set_targets], [Second_set, Second_set_targets], [Third_set, Third_set_targets]]

    # Determine the number of samples from the first set (assumed consistent across all sets)
    n_samples = len(Testing_patterns[0][0])

    # Initialize and load the saved states into model instances
    loaded_model = FirstOrderNetwork(hidden, factor, gelu)
    loaded_model_2 = SecondOrderNetwork(gelu, hidden_2nd)

    loaded_model.load_state_dict(torch.load(PATH))
    loaded_model_2.load_state_dict(torch.load(PATH_2))

    # Ensure the models are moved to the appropriate device (CPU/GPU) and set to evaluation mode
    loaded_model.to(device)
    loaded_model_2.to(device)

    loaded_model.eval()
    loaded_model_2.eval()

    return Testing_patterns, n_samples, loaded_model, loaded_model_2

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


# Function to test the model using the configured testing patterns
def plot_input_output(input_data, output_data, index):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Set figsize to (8, 8) for a square figure

    # Plot input data
    im1 = axes[0].imshow(input_data.cpu().numpy(), aspect='auto', cmap='viridis')
    axes[0].set_title('Input')
    fig.colorbar(im1, ax=axes[0])

    # Plot output data
    im2 = axes[1].imshow(output_data.cpu().numpy(), aspect='auto', cmap='viridis')
    axes[1].set_title('Output')
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle(f'Testing Pattern {index+1}')
    plt.show()
    plt.close(fig)


# Function to test the model using the configured testing patterns
def testing(testing_patterns, n_samples, loaded_model, loaded_model_2,factor , plot, cascade_rate, type_cascade):

    def generate_chance_level(shape):
      chance_level = np.random.rand(*shape).tolist()
      return chance_level

    results_for_plotting = []
    max_values_output_first_order = []
    max_indices_output_first_order = []
    max_values_patterns_tensor = []
    max_indices_patterns_tensor = []
    f1_scores_wager = []

    mse_losses_indices = []
    mse_losses_values = []
    discrimination_performances = []


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

    # Iterate through each set of testing patterns and targets
    for i in range(len(testing_patterns)):
        with torch.no_grad():  # Ensure no gradients are computed during testing

            hidden_representation=None
            output_first_order=None
            comparison_out=None

            #For low vision the stimulus threshold was set to 0.3 as can seen in the generate_patters function
            threshold=0.5
            if i==2:
                threshold=0.15

            # Obtain output from the first order model
            input_data = testing_patterns[i][0]
            for j in range(cascade_iterations_one):
                hidden_representation, output_first_order = loaded_model(input_data, hidden_representation, output_first_order, cascade_rate_one)

            for j in range(cascade_iterations_two):
                output_second_order, comparison_out = loaded_model_2(input_data, output_first_order, comparison_out, cascade_rate_two)

            delta=100*factor

            #print("driscriminator")
            #print((output_first_order[delta:].argmax(dim=1) == input_data[delta:].argmax(dim=1)).to(float).mean())
            discrimination_performance = round((output_first_order[delta:].argmax(dim=1) == input_data[delta:].argmax(dim=1)).to(float).mean().item(), 2)
            discrimination_performances.append(discrimination_performance)


            chance_level = torch.Tensor( generate_chance_level((200*factor,100)))
            discrimination_random= round((chance_level[delta:].argmax(dim=1) == input_data[delta:].argmax(dim=1)).to(float).mean().item(), 2)
            #print("chance level" , discrimination_random)



            #count all patterns in the dataset
            wagers = output_second_order[delta:].cpu()

            _, targets_2 = torch.max(testing_patterns[i][1], 1)
            targets_2 = targets_2[delta:].cpu()

            # Convert targets to binary classification for wagering scenario
            targets_2 = (targets_2 > 0).int()

            # Convert tensors to NumPy arrays for metric calculations
            predicted_np = wagers.numpy().flatten()
            targets_2_np = targets_2.numpy()

            #print("number of targets," , len(targets_2_np))

            #print(predicted_np)
            #print(targets_2_np)

            # Calculate True Positives, True Negatives, False Positives, and False Negatives
            TP = np.sum((predicted_np >  threshold) & (targets_2_np > threshold))
            TN = np.sum((predicted_np <  threshold ) & (targets_2_np < threshold))
            FP = np.sum((predicted_np >  threshold) & (targets_2_np <  threshold))
            FN = np.sum((predicted_np <  threshold) & (targets_2_np >  threshold))

            # Compute precision, recall, F1 score, and accuracy for both high and low wager scenarios
            precision_h, recall_h, f1_score_h, accuracy_h = compute_metrics(TP, TN, FP, FN)

            f1_scores_wager.append(accuracy_h)

            # Collect results for plotting
            results_for_plotting.append({
                "counts": [[TP, FP, TP + FP]],
                "metrics": [[precision_h, recall_h, f1_score_h, accuracy_h]],
                "title_results": f"Results Table - Set {i+1}",
                "title_metrics": f"Metrics Table - Set {i+1}"
            })

            if plot==True:
              # Plot input and output of the first-order network
              plot_input_output(input_data, output_first_order, i)

            max_vals_out, max_inds_out = torch.max(output_first_order[100:], dim=1)
            max_inds_out[max_vals_out == 0] = 0
            max_values_output_first_order.append(max_vals_out.tolist())
            max_indices_output_first_order.append(max_inds_out.tolist())

            max_vals_pat, max_inds_pat = torch.max(input_data[100:], dim=1)
            max_inds_pat[max_vals_pat == 0] = 0
            max_values_patterns_tensor.append(max_vals_pat.tolist())
            max_indices_patterns_tensor.append(max_inds_pat.tolist())


            # Add quadratic fit to scatter plot
            x_indices = max_indices_patterns_tensor[i]
            y_indices = max_indices_output_first_order[i]
            y_pred_indices = perform_quadratic_regression(x_indices, y_indices)

            # Calculate MSE loss for indices
            mse_loss_indices = np.mean((np.array(x_indices) - np.array(y_indices)) ** 2)
            mse_losses_indices.append(mse_loss_indices)

            # Add quadratic fit to scatter plot
            x_values = max_values_patterns_tensor[i]
            y_values = max_values_output_first_order[i]
            y_pred_values = perform_quadratic_regression(x_values, y_values)

            # Calculate MSE loss for values
            mse_loss_values = np.mean((np.array(x_values) - np.array(y_values)) ** 2)
            mse_losses_values.append(mse_loss_values)

            if plot==True:

              fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Set figsize to (10, 10) for a square figure

              # Scatter plot of indices: patterns_tensor vs. output_first_order
              axs[0].scatter(max_indices_patterns_tensor[i], max_indices_output_first_order[i], alpha=0.5)
              axs[0].set_title(f'Stimuli location: Condition {i+1} - First Order Input vs. First Order Output')
              axs[0].set_xlabel('First Order Input Indices')
              axs[0].set_ylabel('First Order Output Indices')
              #axs[0].plot(x_indices, y_pred_indices, color='skyblue')

              # Scatter plot of values: patterns_tensor vs. output_first_order
              axs[1].scatter(max_values_patterns_tensor[i], max_values_output_first_order[i], alpha=0.5)
              axs[1].set_title(f'Stimuli Values: Condition {i+1} - First Order Input vs. First Order Output')
              axs[1].set_xlabel('First Order Input Values')
              axs[1].set_ylabel('First Order Output Values')
              #axs[1].plot(x_values, y_pred_values, color='skyblue')

              plt.tight_layout()
              plt.savefig(f'Blindsight_testing_pattern_{i}.png', bbox_inches='tight')

              plt.show()
              plt.close(fig)


    return f1_scores_wager, mse_losses_indices , mse_losses_values, discrimination_performances, results_for_plotting


def plot_testing(results_seed, discrimination_seed, seeds, title):
    print(results_seed)
    print(discrimination_seed)

    Testing_graph_names = ["Suprathreshold stimulus", "Subthreshold stimulus", "Low Vision"]

    fig, ax = plt.subplots(figsize=(14, len(results_seed[0]) * 2 + 2))  # Adjusted for added header space
    ax.axis('off')
    ax.axis('tight')

    # Define column labels
    col_labels = ["Scenario", "F1 SCORE\n(2nd order network)", "RECALL\n(2nd order network)", "PRECISION\n(2nd order network)", "Discrimination Performance\n(1st order network)", "ACCURACY\n(2nd order network)"]

    # Initialize list to hold all rows of data including headers
    full_data = []

    # Calculate averages and standard deviations
    for i in range(len(results_seed[0])):
        metrics_list = [result[i]["metrics"][0] for result in results_seed]  # Collect metrics for each seed
        discrimination_list = [discrimination_seed[j][i] for j in range(seeds)]

        # Calculate averages and standard deviations for metrics
        avg_metrics = np.mean(metrics_list, axis=0).tolist()
        std_metrics = np.std(metrics_list, axis=0).tolist()

        # Calculate average and standard deviation for discrimination performance
        avg_discrimination = np.mean(discrimination_list)
        std_discrimination = np.std(discrimination_list)

        # Format the row with averages and standard deviations
        row = [
            Testing_graph_names[i],
            f"{avg_metrics[2]:.2f} ± {std_metrics[2]:.2f}",  # F1 SCORE
            f"{avg_metrics[1]:.2f} ± {std_metrics[1]:.2f}",  # RECALL
            f"{avg_metrics[0]:.2f} ± {std_metrics[0]:.2f}",  # PRECISION
            f"{avg_discrimination:.2f} ± {std_discrimination:.2f}",  # Discrimination Performance
            f"{avg_metrics[3]:.2f} ± {std_metrics[3]:.2f}"  # ACCURACY
        ]
        full_data.append(row)

    # Extract metric values for color scaling (excluding the first and last columns which are text)
    metric_values = np.array([[float(x.split(" ± ")[0]) for x in row[1:]] for row in full_data])  # Convert to float for color scaling
    max_value = np.max(metric_values)
    colors = metric_values / max_value  # Normalize for color mapping

    # Prepare colors for all cells, defaulting to white for non-metric cells
    cell_colors = [["white"] * len(col_labels) for _ in range(len(full_data))]
    for i, row in enumerate(colors):
        cell_colors[i][1] = plt.cm.RdYlGn(row[0])
        cell_colors[i][2] = plt.cm.RdYlGn(row[1])
        cell_colors[i][3] = plt.cm.RdYlGn(row[2])
        cell_colors[i][5] = plt.cm.RdYlGn(row[3])  # Adding color for accuracy

    # Adding color for discrimination performance
    discrimination_colors = colors[:, 3]
    for i, dp_color in enumerate(discrimination_colors):
        cell_colors[i][4] = plt.cm.RdYlGn(dp_color)

    # Create the main table with cell colors
    table = ax.table(cellText=full_data, colLabels=col_labels, loc='center', cellLoc='center', cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Set the height of the header row to be double that of the other rows
    for j, col_label in enumerate(col_labels):
        cell = table[(0, j)]
        cell.set_height(cell.get_height() * 2)

    # Add chance level table
    chance_level_data = [["Chance Level\nDiscrimination(1st)", "Chance Level\nAccuracy(2nd)"],
                         ["0.010", "0.50"]]

    chance_table = ax.table(cellText=chance_level_data, bbox=[1.2, 0.5, 0.3, 0.1], cellLoc='center', colWidths=[0.1, 0.1])
    chance_table.auto_set_font_size(False)
    chance_table.set_fontsize(10)
    chance_table.scale(2.2, 2.2)

    # Set the height of the header row to be double that of the other rows in the chance level table
    for j in range(len(chance_level_data[0])):
        cell = chance_table[(0, j)]
        cell.set_height(cell.get_height() * 2)

    plt.title(title, pad=20, fontsize=16)
    plt.show()
    plt.close(fig)


# %% [markdown]
# # MAIN CODE - TRAINING DEFINITION

# %%
def train(hidden, hidden_2nd, factor, gelu, stepsize, gam, meta, optimizer, seeds, noise_level, plotting, cascade_rate, type_cascade):
    """
    Trains the Blindsight model which consists of a first-order network (autoencoder) and
    a second-order network (metacognitive wagering component).
    
    Parameters:
    - hidden: Size of hidden layers in the first-order network
    - hidden_2nd: Size of hidden layers in the second-order network
    - factor: Scaling factor for model parameters or losses
    - gelu: Boolean flag to use GELU activation instead of default
    - stepsize: Learning rate step size
    - gam: Gamma parameter for learning rate scheduler
    - meta: Metacognition parameter controlling influence of second-order network
    - optimizer: Type of optimizer to use (e.g., 'adam', 'sgd')
    - seeds: Number of random seeds to run training with
    - noise_level: Amount of noise to add during training
    - plotting: Boolean flag to enable visualization of training progress
    - cascade_rate: Rate parameter for the cascade model's accumulation
    - type_cascade: Type of cascade model implementation to use
    
    Returns:
    - Various performance metrics and data for analysis and visualization
    """
    
    # Initialize lists to store performance metrics for violin plots
    # Lists for first-order network performance across 3 test conditions
    list_violin_1st = [[], [], []]
    # Lists for second-order network (wagering) performance across 3 test conditions
    list_violin_2nd = [[], [], []]
    
    # Lists to collect discrimination performance and results across all seeds
    list_discr_perf = []
    list_results = []
    all_f1_scores_wager = []
    all_discrimination_performances = []
    
    # Run training with different random seeds for statistical robustness
    for seed in range(seeds):
        # Only plot for the first seed to avoid excessive visualizations
        if plotting == True and seed > 0:
            plotting = False
        
        # Set up networks, loss criteria, optimizers and schedulers
        # First-order network: autoencoder for pattern recognition
        # Second-order network: metacognitive component with wagering units
        first_order_network, second_order_network, criterion_1, criterion_2, optimizer_1, optimizer_2, scheduler_1, scheduler_2 = prepare_pre_training(
            hidden, hidden_2nd, factor, gelu, stepsize, gam, optimizer
        )
        
        # Pre-train both networks using contrastive loss (1st) and cross-entropy (2nd)
        # The cascade model is applied here for graded activation accumulation
        first_order_network_pre, second_order_network_pre, epoch_1_order, epoch_2_order, max_values_indices = pre_train(
            first_order_network, second_order_network, criterion_1, criterion_2, 
            optimizer_1, optimizer_2, scheduler_1, scheduler_2, factor, meta, 
            noise_level, cascade_rate, type_cascade
        )
        
        # Set up the testing patterns and load the pre-trained models
        # For Blindsight, this includes suprathreshold patterns and 3 types of test patterns
        Testing_patterns, n_samples, loaded_model, loaded_model_2 = config_training(
            first_order_network_pre, second_order_network_pre, hidden, hidden_2nd, 
            factor, gelu, noise_level
        )
        
        # Test the models on different pattern types and evaluate performance
        # Records discrimination performance (1st-order) and F1 scores for wagering (2nd-order)
        f1_scores_wager, mse_losses_indices, mse_losses_values, discrimination_performances, results_for_plotting = testing(
            Testing_patterns, n_samples, loaded_model, loaded_model_2, factor, 
            plotting, cascade_rate, type_cascade
        )
        
        # Calculate the improvement in loss during training (convergence metric)
        loss_slope = epoch_1_order[0] - epoch_1_order[-1]
        # Calculate mean MSE for reconstructed indices and values
        mean_mse_indices = np.mean(mse_losses_indices)
        mean_mse_values = np.mean(mse_losses_values)
        
        # Commented-out debugging print statement with detailed parameters and results
        # Would show configuration parameters and performance metrics for each run
        
        # Collect performance metrics for this seed
        all_f1_scores_wager.append(f1_scores_wager)
        all_discrimination_performances.append(discrimination_performances)
        list_discr_perf.append(discrimination_performances)
        list_results.append(results_for_plotting)
        
        # Organize data for violin plots by test condition type
        # First-order network discrimination performance for each test type
        list_violin_1st[0].append(discrimination_performances[0])  # Test type 1
        list_violin_1st[1].append(discrimination_performances[1])  # Test type 2
        list_violin_1st[2].append(discrimination_performances[2])  # Test type 3
        
        # Second-order network wagering F1 scores for each test type
        list_violin_2nd[0].append(f1_scores_wager[0])  # Test type 1
        list_violin_2nd[1].append(f1_scores_wager[1])  # Test type 2
        list_violin_2nd[2].append(f1_scores_wager[2])  # Test type 3
    
    # Compute average performance metrics across all seeds
    average_f1_scores_wager = np.mean(all_f1_scores_wager, axis=0)
    average_discrimination_performances = np.mean(all_discrimination_performances, axis=0)
    
    # Compute standard deviations of performance metrics
    average_std_scores_wager = np.std(all_f1_scores_wager, axis=0)
    average_std_discrimination_performances = np.std(all_discrimination_performances, axis=0)
    
    # Organize data for visualization plots
    plot_data = {
        'pre_train': (epoch_1_order, epoch_2_order, max_values_indices),
        'test': (list_results, list_discr_perf, seeds)
    }
    
    # Return all performance metrics and plotting data
    return (
        average_f1_scores_wager, average_std_scores_wager, epoch_1_order,
        mean_mse_indices, mean_mse_values, average_discrimination_performances,
        average_std_discrimination_performances, plot_data, list_violin_1st, list_violin_2nd
    )
    
def plots(plot_data , title):
  #plot the results of a trained and tested network
  pre_train_plots(*plot_data['pre_train'] , title)
  plot_testing(*plot_data['test'] , title)


# %%


def initialize_global():
    global Input_Size_1, Hidden_Size_1, Output_Size_1, Input_Size_2
    global num_units, patterns_number
    global learning_rate_1, learning_rate_2, n_epochs, momentum, temperature , Threshold
    global First_set, Second_set, Third_set
    global First_set_targets, Second_set_targets, Third_set_targets
    global epoch_list, epoch_1_order, epoch_2_order, patterns_matrix1
    global testing_graph_names

    # Network sizes
    Input_Size_1 = 100
    Hidden_Size_1 = 60
    Output_Size_1 = 100
    Input_Size_2 = 100

    # Patterns
    num_units = 100
    patterns_number = 200

    # Pre-training and hyperparameters
    learning_rate_1 = 0.5
    learning_rate_2 = 0.1
    n_epochs = 200
    momentum = 0.9
    temperature = 1.0
    Threshold=0.5

    # Testing
    First_set = []
    Second_set = []
    Third_set = []
    First_set_targets = []
    Second_set_targets = []
    Third_set_targets = []

    # Graphic of pretraining
    epoch_list = list(range(1, n_epochs + 1))
    epoch_1_order = np.zeros(n_epochs)
    epoch_2_order = np.zeros(n_epochs)
    patterns_matrix1 =  torch.zeros((n_epochs, patterns_number), device=device)  # Initialize patterns_matrix as a PyTorch tensor on the GPU


def title(string):
    # Split the string into multiple lines if necessary
    words = string.split()
    max_words_per_line = len(words) // 3 if len(words) > 2 else len(words)

    lines = [' '.join(words[i:i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)]

    # Plot the title of the currently trained model, inside a rectangle
    fig, ax = plt.subplots()
    rectangle = patches.Rectangle((0.05, 0.1), 0.9, 0.4, linewidth=1, edgecolor='r', facecolor='blue', alpha=0.5)
    ax.add_patch(rectangle)

    # Position the text over multiple lines
    y_positions = [0.4, 0.3, 0.2]
    for i, line in enumerate(lines):
        plt.text(0.5, y_positions[i], line, horizontalalignment='center', verticalalignment='center', fontsize=26, color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Compare your results with the patterns generate below
initialize_global()
set_1, _ = create_patterns(0,1, 0.0012)
set_2, _ = create_patterns(1,1, 0.0012)
set_3, _ = create_patterns(2,1, 0.0012)

# Plot
plot_signal_max_and_indicator(set_1.detach().cpu(), "Suprathreshold dataset")
plot_signal_max_and_indicator(set_2.detach().cpu(), "Subthreshold dataset")
plot_signal_max_and_indicator(set_3.detach().cpu(), "Low Vision dataset")


# Plotting function for scaling
def plot_scaling(scaling_metric, f1_scores_list, discr_perf_list, std_wager, std_discrimination, f1_scores_list_meta, discr_perf_list_meta, std_wager_meta, std_discrimination_meta, x_axis):
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
    
    scenarios = ['Suprathreshold', 'Subthreshold', 'Low Vision']
    scenarios_meta =  ['Suprathreshold(Maps)', 'Subthreshold(Maps)', 'Low Vision(Maps)']

    # Plot F1 Scores
    for i, scenario in enumerate(scenarios):
        axes[0].plot(scaling_metric, f1_scores_array[i], label=scenario)
        axes[0].fill_between(scaling_metric, f1_scores_array[i] - std_wager_array[i], f1_scores_array[i] + std_wager_array[i], alpha=0.2)
    
    # Plot F1 Scores
    for i, scenario in enumerate(scenarios_meta):
        axes[0].plot(scaling_metric, f1_scores_array_meta[i], label=scenario)
        axes[0].fill_between(scaling_metric, f1_scores_array_meta[i] - std_wager_array_meta[i], f1_scores_array_meta[i] + std_wager_array_meta[i], alpha=0.2)
        
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
    plt.savefig('NeurIPS_v2_Blindsight_scaling_{}.png'.format(x_axis.replace(" ", "_").replace("/", "_")), bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_violin(list_violin_1_supra, list_violin_1_sub, list_violin_1_low,
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
    plt.savefig('IJCAI_violin_plot_Blindsight_with_table_{}.png'.format(plot_title.replace(" ", "_").replace("/", "_")), bbox_inches='tight')
    plt.show()
    plt.close(fig)


def convert_arrays_to_lists(data):
    """Recursively converts NumPy arrays to Python lists."""
    if isinstance(data, list):
        return [convert_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
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




def load_scaling_data_from_csv(filename):
    """Loads scaling data from a CSV file"""
    data = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # Convert string representations of lists back to actual lists
            data.append([eval(col) if i < len(row)-1 else col for i, col in enumerate(row)])
    return data

def plot_scaling_discrimination(scaling_factors, discrimination_data, factor_second_values, settings_data, std_data=None):
    """
    Plots discrimination performance with standard deviation bands for different secondary factors across multiple settings
    
    Args:
        scaling_factors: List of primary scaling factors (x-axis)
        discrimination_data: Dictionary where keys are settings (1-6) and values are nested lists
                          where each sublist contains discrimination values for a specific
                          secondary factor across all primary factors
        factor_second_values: List of secondary factor values (for legend)
        settings_data: List of settings to plot (1-6)
        std_data: Dictionary with same structure as discrimination_data, containing standard deviation values
    """
    # Create figure with 6 subplots (2 columns, 3 rows)
    fig, axs = plt.subplots(3, 2, figsize=(16, 18), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(factor_second_values)))
    
    # For each setting (1-6)
    for setting_idx, setting in enumerate(settings_data):
        ax = axs[setting_idx]
        
        # Get data for this setting
        setting_data = discrimination_data.get(setting, [[] for _ in range(len(factor_second_values))])
        
        # Get standard deviation data for this setting if available
        setting_std_data = None
        if std_data is not None:
            setting_std_data = std_data.get(setting, [[] for _ in range(len(factor_second_values))])
        
        # Plot each secondary factor line
        for i, factor_second in enumerate(factor_second_values):
            # Plot the main line
            ax.plot(
                scaling_factors, 
                setting_data[i],
                marker='o',
                linestyle='-',
                color=colors[i],
                label=f'Factor = {factor_second}'
            )
            
            # Add standard deviation bands if available
            if setting_std_data is not None and len(setting_std_data[i]) > 0:
                # Calculate upper and lower bounds
                upper_bound = [d + s for d, s in zip(setting_data[i], setting_std_data[i])]
                lower_bound = [d - s for d, s in zip(setting_data[i], setting_std_data[i])]
                
                # Plot the filled area between upper and lower bounds
                ax.fill_between(
                    scaling_factors,
                    lower_bound,
                    upper_bound,
                    color=colors[i],
                    alpha=0.2  # Transparency
                )
        
        ax.set_xscale('log', base=2)  # Log scale for x-axis
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
    fig.text(0.08, 0.5, 'Discrimination Performance', va='center', rotation='vertical', fontsize=14)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(factor_second_values), fontsize=12, frameon=True)
    
    plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.92])  # Adjust layout to make room for common labels
    
    # Add overall title
    plt.suptitle('Discrimination Performance by Scaling Factor Across Settings', fontsize=16, y=0.995)
    
    # Save the plot
    plt.savefig(f'blindsight_discrimination_scaling_plot_all_settings.png', dpi=300)
    plt.show()
    plt.close(fig)
    print(f"plot saved as discrimination_scaling_plot_all_settings.png'.")

 
# Function to run a single setting experiment and collect data
def run_setting_experiment(setting, scaling_factors, factors_second, default_hidden_first, default_hidden_second, seeds_violin, cascade_off, cascade_mode  ):
    """
    Run experiments for a single setting and return the collected data
    """
    # Initialize data structure for this setting
    discrimination_data = [[] for _ in range(len(factors_second))]
    f1_scores_data = [[] for _ in range(len(factors_second))]
    std_discrimination_data = [[] for _ in range(len(factors_second))]
    std_wager_data = [[] for _ in range(len(factors_second))]
    
    # For each primary scaling factor
    for p_idx, factor in enumerate(scaling_factors):
        # For each secondary scaling factor
        for s_idx, factor_second in enumerate(factors_second):
            initialize_global()
            
            hidden_first_scaling = int(default_hidden_first)
            hidden_second_scaling = int(default_hidden_first*factor_second)
            data_factor=factor

            # Select the appropriate training function based on setting
            if setting == 1:
                f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_2nd=hidden_second_scaling, 
                    factor=data_factor,
                    gelu=False, 
                    stepsize=25, 
                    gam=0.98, 
                    meta=False, 
                    optimizer='ADAMAX', 
                    seeds=seeds_violin, 
                    noise_level=0.0012, 
                    plotting=False, 
                    cascade_rate=cascade_off, 
                    type_cascade=4
                )
            elif setting == 2:
                f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_2nd=hidden_second_scaling, 
                    factor=data_factor, 
                    gelu=False, 
                    stepsize=25, 
                    gam=0.98,
                    meta=False, 
                    optimizer='ADAMAX', 
                    seeds=seeds_violin, 
                    noise_level=0.0012,
                    plotting=False, 
                    cascade_rate=cascade_mode, 
                    type_cascade=2
                )
            elif setting == 3:
                f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_2nd=hidden_second_scaling, 
                    factor=data_factor, 
                    gelu=False, 
                    stepsize=25, 
                    gam=0.98,
                    meta=True, 
                    optimizer='ADAMAX', 
                    seeds=seeds_violin, 
                    noise_level=0.0012,
                    plotting=False, 
                    cascade_rate=cascade_off, 
                    type_cascade=4
                )
            elif setting == 4:
                f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_2nd=hidden_second_scaling, 
                    factor=data_factor, 
                    gelu=False, 
                    stepsize=25, 
                    gam=0.98,
                    meta=True, 
                    optimizer='ADAMAX', 
                    seeds=seeds_violin, 
                    noise_level=0.0012,
                    plotting=False, 
                    cascade_rate=cascade_mode, 
                    type_cascade=2
                )
            elif setting == 5:
                f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_2nd=hidden_second_scaling, 
                    factor=data_factor, 
                    gelu=False, 
                    stepsize=25, 
                    gam=0.98,
                    meta=True, 
                    optimizer='ADAMAX', 
                    seeds=seeds_violin, 
                    noise_level=0.0012,
                    plotting=False, 
                    cascade_rate=cascade_mode, 
                    type_cascade=3
                )
            elif setting == 6:
                f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
                    hidden=hidden_first_scaling, 
                    hidden_2nd=hidden_second_scaling, 
                    factor=data_factor, 
                    gelu=False, 
                    stepsize=25, 
                    gam=0.98,
                    meta=True, 
                    optimizer='ADAMAX', 
                    seeds=seeds_violin, 
                    noise_level=0.0012,
                    plotting=False, 
                    cascade_rate=cascade_mode, 
                    type_cascade=1
                )
            
            # Store results by secondary factor
            discrimination_data[s_idx].append(discrimination_performances[0])
            f1_scores_data[s_idx].append(f1_scores_wager[0])
            std_discrimination_data[s_idx].append(std_discrimination[0])
            std_wager_data[s_idx].append(std_wager[0])
            
            # Save individual experiment result
            experiment_log = {
                'primary_factor': factor,
                'secondary_factor': factor_second,
                'f1_score': f1_scores_wager,
                'std_wager': std_wager,
                'discrimination_performance': discrimination_performances[0],
                'std_discrimination': std_discrimination[0],
                'meta': setting >= 3,  # settings 3-6 use meta=True
                'cascade_type': [4, 2, 4, 2, 3, 1][setting-1],  # Map setting to cascade type
                'hidden_first': default_hidden_first*factor,
                'hidden_second': default_hidden_first*factor*factor_second
            }
            
            # Save individual experiment details
            log_filename = f'experiment_factor_{factor}_secondary_{factor_second}_setting_{setting}.csv'
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
        'discrimination_data': convert_arrays_to_lists(discrimination_data),
        'f1_scores_data': convert_arrays_to_lists(f1_scores_data),
        'std_discrimination_data': convert_arrays_to_lists(std_discrimination_data),
        'std_wager_data': convert_arrays_to_lists(std_wager_data),
        'setting': setting
    }
    
    # Save setting results to CSV
    log_filename = f'Blindsight_scaling_plot_data_setting_{setting}.csv'
    with open(log_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(setting_data.keys())
        
        # Write the data (one row)
        writer.writerow([convert_arrays_to_lists(val) if isinstance(val, (list, np.ndarray)) 
                        else val for val in setting_data.values()])
    
    print(f"Setting {setting} data saved to '{log_filename}'.")
    
    return discrimination_data, f1_scores_data, std_discrimination_data, std_wager_data

# Loading data back from CSV for verification or later use
def load_and_plot_from_csv(filename):
    """Loads data from CSV and creates the discrimination plot"""
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        row = next(reader)
        
        # Convert string representations back to lists/values
        data = {header[i]: eval(row[i]) if i < len(row) else None for i in range(len(header))}
        
        # Create plot from loaded data
        plot_scaling_discrimination(
            data['scaling_factors'],
            data['discrimination_data'],
            data['factors_second'],
            data['setting']
        )
    
    return data

    
def load_setting_data_from_csv(setting):
    """
    Load experiment data for a specific setting from CSV file
    Returns a dictionary with the loaded data
    """
    filename = f'Blindsight_scaling_plot_data_setting_{setting}.csv'
    
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found")
        return None
    
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        keys = next(reader)  # Read header row
        values = next(reader)  # Read data row
        
        # Convert string representations back to Python objects
        data = {}
        for i, key in enumerate(keys):
            try:
                # Parse string representations of lists back to actual lists
                if key in ['scaling_factors', 'factors_second', 'setting']:
                    data[key] = ast.literal_eval(values[i])
                else:
                    # For the nested data structures
                    data[key] = ast.literal_eval(values[i])
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing {key}: {e}")
                data[key] = values[i]  # Keep as string if parsing fails
    
    return data
def main():
    # Hyperparameter configuration for model tuning
    hidden_sizes = [30, 40, 50, 60, 100]  # Hidden layer sizes for first-order network
    factors = [1]  # Data multiplication factor (10 = 10x data amount from the original paper)
    gelus = [False]  # Activation function selection: True = GELU, False = ReLU
    step_sizes = [12, 25, 50]  # Learning rate scheduler step size (epochs between LR updates)
    gammas = [0.98, 0.99]  # Learning rate decay factor applied every step_size epochs
    metalayers = [True]  # Enable/disable second-order metacognitive network
    
    # Additional configuration parameters
    loss_function = []  # Not used but prepared for potential loss function variations
    optimizer = ['ADAMAX']  # Optimizer choice
    seeds = 5  # Number of random seeds for basic experiments
    seeds_violin = 450  # Number of seeds for violin plot experiments (statistical significance)
    
    # Cascade model parameters
    cascade_mode = 0.02  # Cascade model rate parameter
    # cascade_mode = 0.015625  # Alternative cascade rate (commented out)
    # cascade_mode = 0.01  # Alternative cascade rate (commented out)
    cascade_off = 1.0  # Value to effectively disable cascade mode
    
    num_iterations = 1  # Number of training runs per configuration
    
    # Variables to track best performing model configuration
    best_optimization_variable = 0
    best_plot_data = None
    best_f1_wager = None
    best_1st_loss = None
    best_hidden_size = None
    best_factor = None
    best_activation = None
    best_step_size = None
    best_gamma = None
    best_metalayer = None
    best_optimizer = None
    
    # Generate all combinations of hyperparameters for grid search
    hyperparameters = list(product(hidden_sizes, factors, gelus, step_sizes, gammas, metalayers, optimizer))
    
    # Execution mode flags
    Training = False  # Run training experiments
    
    # Cascade type explanations:
    # cascade type 1: both 1st and 2nd order networks use cascade mode
    # cascade type 2: only 1st order network uses cascade model
    # cascade type 3: only 2nd order network uses cascade model
    # cascade type 4: neither network uses cascade model (baseline)
    
    if Training:
        # Configuration 1: Baseline model - No cascade, No 2nd-order network
        initialize_global()  # Reset global variables
        f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, \
        std_discrimination, plot_data, list_violin_1st, list_violin_2nd = train(
            hidden=40, hidden_2nd=100, factor=1, gelu=False, stepsize=25, gam=0.98, 
            meta=False, optimizer='ADAMAX', seeds=seeds_violin, noise_level=0.0012, 
            plotting=False, cascade_rate=cascade_off, type_cascade=4
        )
        
        # Configuration 2: Only cascade model on 1st-order network
        initialize_global()
        f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, \
        std_discrimination, plot_data, list_violin_1st_conf2, list_violin_2nd_conf2 = train(
            hidden=40, hidden_2nd=100, factor=1, gelu=False, stepsize=25, gam=0.98, 
            meta=False, optimizer='ADAMAX', seeds=seeds_violin, noise_level=0.0012, 
            plotting=False, cascade_rate=cascade_mode, type_cascade=2
        )
        
        # Configuration 3: Only 2nd-order network, no cascade
        initialize_global()
        f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, \
        std_discrimination, plot_data, list_violin_1st_conf3, list_violin_2nd_conf3 = train(
            hidden=40, hidden_2nd=100, factor=1, gelu=False, stepsize=25, gam=0.98, 
            meta=True, optimizer='ADAMAX', seeds=seeds_violin, noise_level=0.0012, 
            plotting=False, cascade_rate=cascade_off, type_cascade=4
        )
        
        # Configuration 4: 2nd-order network + cascade on 1st-order network
        initialize_global()
        f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, \
        std_discrimination, plot_data, list_violin_1st_conf4, list_violin_2nd_conf4 = train(
            hidden=40, hidden_2nd=100, factor=1, gelu=False, stepsize=25, gam=0.98, 
            meta=True, optimizer='ADAMAX', seeds=seeds_violin, noise_level=0.0012, 
            plotting=False, cascade_rate=cascade_mode, type_cascade=2
        )
        
        # Configuration 5: 2nd-order network + cascade on 2nd-order network only
        initialize_global()
        f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, \
        std_discrimination, plot_data, list_violin_1st_conf5, list_violin_2nd_conf5 = train(
            hidden=40, hidden_2nd=100, factor=1, gelu=False, stepsize=25, gam=0.98, 
            meta=True, optimizer='ADAMAX', seeds=seeds_violin, noise_level=0.0012, 
            plotting=False, cascade_rate=cascade_mode, type_cascade=3
        )
        
        # Configuration 6: Full model - 2nd-order network + cascade on both networks
        initialize_global()
        f1_scores_wager, std_wager, epoch_1_loss, mse_indices, mse_values, discrimination_performances, \
        std_discrimination, plot_data, list_violin_1st_cascade, list_violin_2nd_cascade = train(
            hidden=40, hidden_2nd=100, factor=1, gelu=False, stepsize=25, gam=0.98, 
            meta=True, optimizer='ADAMAX', seeds=seeds_violin, noise_level=0.0012, 
            plotting=False, cascade_rate=cascade_mode, type_cascade=1
        )
        
        # Prepare data for violin plots to visualize statistical distributions
        log_data_violin = [
            # First row: Discrimination performance (main task) for all configurations
            ['Discrimination Performance (main task)', 
             list_violin_1st[0], list_violin_1st_conf2[0], list_violin_1st_conf3[0], 
             list_violin_1st_conf4[0], list_violin_1st_conf5[0], list_violin_1st_cascade[0]],
            # Second row: Wagering accuracy (metacognitive task) for all configurations
            ['Accuracy (wagering)', 
             list_violin_2nd[0], list_violin_2nd_conf2[0], list_violin_2nd_conf3[0], 
             list_violin_2nd_conf4[0], list_violin_2nd_conf5[0], list_violin_2nd_cascade[0]]
        ]
        
        # Save experimental results to CSV for further analysis and visualization
        with open('Blindsight_violin_plot_logs_conference.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header row
            writer.writerow(['Title', 'List 1', 'List 2', 'List 3', 'Cascade List 1', 'Cascade List 2', 'Cascade List 3'])
            
            # Write the experimental data
            writer.writerows(log_data_violin)
        
        print("Violin plot variables saved to 'Blindsight_violin_plot_logs_conference.csv'.")
    
        # Load the saved experimental data
        violin_data = load_violin_data_from_csv('Blindsight_violin_plot_logs_conference.csv')
        
        # Generate violin plots for visual comparison of all configurations
        for data in violin_data:
            title, list1, list2, list3, cascade_list1, cascade_list2, cascade_list3 = data
            plot_violin(list1, list2, list3, cascade_list1, cascade_list2, cascade_list3, title, 1)

    default_hidden_first=40
    default_hidden_second=100
    #scaling_factors= [1, 2, 5 , 10,  15 , 25 , 50 ]
    scaling_factors= [1, 5 ,  15 , 25 , 50 ]

    #factors_second= [ 0.1, 0.2, 0.5, 1.0, 2.0]
    factors_second= [ 0.1, 1.0]
    
    seeds_scaling = 5 # Number of seeds for scaling experiments

    setting=1
    scaling=True
    
    load_data=True
    
    if scaling:
        # Dictionary to store results for all settings
        all_discrimination_data = {}
        all_f1_scores_data = {}
        all_std_discrimination_data = {}
        all_std_wager_data = {}
        
        # Run experiments for all 6 settings
        for setting in range(1, 7):
            if load_data:
                print(f"Loading data for setting {setting}")
                # Load data from CSV
                setting_data = load_setting_data_from_csv(setting)
                
                # Store data in respective dictionaries
                all_discrimination_data[setting] = setting_data['discrimination_data']
                all_f1_scores_data[setting] = setting_data['f1_scores_data']
                all_std_discrimination_data[setting] = setting_data['std_discrimination_data']
                all_std_wager_data[setting] = setting_data['std_wager_data']
            else:
                print(f"Starting experiments for setting {setting}")
                discr_data, f1_data, std_discr_data, std_wager_data = run_setting_experiment(
                    setting, scaling_factors, factors_second, default_hidden_first, 
                    default_hidden_second, seeds_scaling, cascade_off, cascade_mode
                )
                
                # Store results
                all_discrimination_data[setting] = discr_data
                all_f1_scores_data[setting] = f1_data
                all_std_discrimination_data[setting] = std_discr_data
                all_std_wager_data[setting] = std_wager_data
        
        # Create combined plots with data from all settings
        plot_scaling_discrimination(
            scaling_factors,
            all_discrimination_data,
            factors_second,
            list(range(1, 7)),  # Settings 1-6
            std_data=all_std_discrimination_data  # Add the standard deviation data
        )
        
        # Save combined results if not loading from CSV
        if not load_data:
            combined_results = {
                'scaling_factors': scaling_factors,
                'factors_second': factors_second,
                'all_discrimination_data': convert_arrays_to_lists(all_discrimination_data),
                'all_f1_scores_data': convert_arrays_to_lists(all_f1_scores_data),
                'all_std_discrimination_data': convert_arrays_to_lists(all_std_discrimination_data),
                'all_std_wager_data': convert_arrays_to_lists(all_std_wager_data)
            }
            
            # Save to CSV
            log_filename = 'Blindsight_scaling_plot_data_all_settings.csv'
            with open(log_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(combined_results.keys())
                
                # Write the data (one row)
                writer.writerow([convert_arrays_to_lists(val) if isinstance(val, (list, dict, np.ndarray))
                            else val for val in combined_results.values()])
            
            print(f"Combined results for all settings saved to '{log_filename}'.")
                    

main()