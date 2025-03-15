"""
Models for ocean vertical mixing prediction.

This module provides neural network models for predicting vertical mixing coefficients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OceanMixingNN(nn.Module):
    """
    Neural network model for predicting ocean vertical mixing shape functions.
    
    Architecture:
    - Input: 4 physical parameters (Coriolis, buoyancy flux, friction velocity, boundary layer depth)
    - Hidden layers: 2 fully connected layers with ReLU activation
    - Output: 16 values representing the shape function at different sigma levels
    """
    
    def __init__(self, in_nodes=4, hidden_nodes=32, out_nodes=16, dropout_rate=0.25):
        """
        Initialize the ocean mixing neural network
        
        Args:
            in_nodes: Number of input features (default 4)
            hidden_nodes: Number of nodes in hidden layers (default 32)
            out_nodes: Number of output nodes (default 16, one per sigma level)
            dropout_rate: Dropout probability for regularization (default 0.25)
        """
        super(OceanMixingNN, self).__init__()
        
        # Define layers
        self.linear1 = nn.Linear(in_nodes, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = nn.Linear(hidden_nodes, out_nodes)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor with shape (batch_size, in_nodes)
            
        Returns:
            Output tensor with shape (batch_size, out_nodes)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return self.linear3(x)


class OceanMixingResidualNN(nn.Module):
    """
    Neural network with residual connections for ocean vertical mixing prediction.
    
    This model adds residual (skip) connections between layers to help with training
    deeper networks.
    """
    
    def __init__(self, in_nodes=4, hidden_nodes=32, out_nodes=16, dropout_rate=0.25, n_blocks=2):
        """
        Initialize the residual neural network
        
        Args:
            in_nodes: Number of input features (default 4)
            hidden_nodes: Number of nodes in hidden layers (default 32)
            out_nodes: Number of output nodes (default 16, one per sigma level)
            dropout_rate: Dropout probability for regularization (default 0.25)
            n_blocks: Number of residual blocks (default 2)
        """
        super(OceanMixingResidualNN, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(in_nodes, hidden_nodes)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ResidualBlock(hidden_nodes, dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_nodes, out_nodes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor with shape (batch_size, in_nodes)
            
        Returns:
            Output tensor with shape (batch_size, out_nodes)
        """
        # Input layer
        x = F.relu(self.input_layer(x))
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output layer
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """
    Residual block with two fully connected layers and a skip connection.
    """
    
    def __init__(self, hidden_nodes, dropout_rate=0.25):
        """
        Initialize the residual block
        
        Args:
            hidden_nodes: Number of nodes in hidden layers
            dropout_rate: Dropout probability for regularization
        """
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        """
        Forward pass through the residual block
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with same shape as input
        """
        identity = x
        
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        
        # Skip connection
        out += identity
        out = F.relu(out)
        
        return out