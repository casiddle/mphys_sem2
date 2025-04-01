import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import os
import random
import hiddenlayer as hl

# Neural Network parameters
DATA_FILE_PATH="Processed_Data/data_sets/big_scan_correct.csv"
EPOCHS = 800
PATIENCE = 50
BATCH_NO = 30
NO_HIDDEN_LAYERS = 10
LEARNING_RATE = 1e-3
NO_NODES = 36
TEST_AND_VALIDATION_SIZE = 0.2
TEST_SIZE = TEST_AND_VALIDATION_SIZE/2
DROPOUT = 0.1
ACTIVATION_FUNCTION = "Leaky ReLU"
PREDICTED_FEATURES = ["Emittance", "Beam Energy", "Beam Spread"]
PREDICTOR_FEATURES = ["X-ray Mean Radiation Radius",'X-ray Critical Energy', 'X-ray Percentage']
INPUT_SIZE = len(PREDICTOR_FEATURES)
TRAIN_TEST_SEED = random.randint(1,300)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=NO_NODES, num_hidden_layers=NO_HIDDEN_LAYERS, num_outputs=len(PREDICTED_FEATURES),dropout_rate=DROPOUT):
        super().__init__()
        
        # Initialize an empty list to hold layers
        layers = []
        
        # First hidden layer (input layer to the first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size)) #Normalisation
        #layers.append(nn.ReLU())  # ReLU activation function
        layers.append(nn.LeakyReLU(negative_slope=0.01))  
        
        # Loop to add the hidden layers
        for _ in range(num_hidden_layers - 1):  # Subtract 1 since the first hidden layer is already added
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size)) 
            #layers.append(nn.ReLU())  # ReLU activation for each hidden layer
            layers.append(nn.LeakyReLU(negative_slope=0.01))  
            layers.append(nn.Dropout(dropout_rate))  # Dropout layer after activation
        
        # Output layer
        layers.append(nn.Linear(hidden_size, num_outputs))  # Output layer (single output)
        #layers.append(nn.ReLU())
        layers.append(nn.LeakyReLU(negative_slope=0.01))  
        #layers.append(nn.Softplus())  

        
        # Use Sequential to combine layers
        self.linear_relu_stack = nn.Sequential(*layers)
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    


































































