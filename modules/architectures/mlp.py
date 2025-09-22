#!/usr/bin/env python

import torch.nn as nn
from utils.activation_function_utils import get_activation_function

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, layer_normalization: bool = False,
                 activation_function_names: list = None, output_activation_function_name: str = None,
                 dropout_rates: list = None, output_dropout_rate: float = 0.0, dropout_at_inference: bool = False):
        """
        Initialize MLP with flexible architecture.

        Args:
            input_size: Size of input layer
            hidden_sizes: Sizes of hidden layers
            output_size: Size of output layer
            layer_normalization: Whether to apply layer normalization
            activation_function_names: Activation functions for hidden layers
            output_activation_function_name: Activation for output layer
            dropout_rates: Dropout rates for each layer
            output_dropout_rate: Dropout rate for output
            dropout_at_inference: Whether to use dropout during inference
        """
        super(MLP, self).__init__()

        assert input_size > 0, "Input size must be positive"
        assert hidden_sizes, "Must specify hidden layer sizes"
        assert output_size > 0, "Output size must be positive"
        if activation_function_names is None:
            activation_function_names = ['leaky_relu'] * len(hidden_sizes)
        elif len(activation_function_names) != len(hidden_sizes):
            print(f"WARNING: Number of hidden layers ({len(hidden_sizes)}) not equal to number of activation functions ({len(activation_function_names)}).")
            activation_function_names = ['leaky_relu'] * len(hidden_sizes)

        if dropout_rates is None:
            dropout_rates = [0.10] * (len(hidden_sizes) + 1)
        elif len(dropout_rates) != len(hidden_sizes) + 1:
            print(f"WARNING: Number of dropout rates ({len(dropout_rates)}) not equal to combined number of input and hidden layers ({len(hidden_sizes) + 1}).")
            dropout_rates = [0.10] * (len(hidden_sizes) + 1)

        self.hidden_sizes = hidden_sizes
        self.layer_normalization = layer_normalization
        self.dropout_at_inference = dropout_at_inference
        self.output_dropout_rate = output_dropout_rate

        layers = []

        # Input layer
        if dropout_rates[0] > 0:
            layers.append(nn.Dropout(p=dropout_rates[0]))
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if activation_function_names[0] == "sigmoid" or activation_function_names[0] == "tanh":
            nn.init.xavier_normal_(layers[-1].weight)
        else:
            nn.init.kaiming_normal_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if self.layer_normalization:
            layers.append(nn.LayerNorm(hidden_sizes[0]))
        activation_function = get_activation_function(activation_function_names[0])
        if activation_function is not None:
            layers.append(activation_function)
        if dropout_rates[1] > 0:
            layers.append(nn.Dropout(p=dropout_rates[1]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if activation_function_names[i] == "sigmoid" or activation_function_names[i] == "tanh":
                nn.init.xavier_normal_(layers[-1].weight)
            else:
                nn.init.kaiming_normal_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            if self.layer_normalization:
                layers.append(nn.LayerNorm(hidden_sizes[i]))
            activation_function = get_activation_function(activation_function_names[i])
            if activation_function is not None:
                layers.append(activation_function)
            if dropout_rates[i+1] > 0:
                layers.append(nn.Dropout(p=dropout_rates[i+1]))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if activation_function_names[-1] == "sigmoid" or activation_function_names[-1] == "tanh":
            nn.init.xavier_normal_(layers[-1].weight)
        else:
            nn.init.kaiming_normal_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        # Optional output activation
        if output_activation_function_name is not None:
            if self.layer_normalization:
                layers.append(nn.LayerNorm(output_size))
            output_activation = get_activation_function(output_activation_function_name)
            layers.append(output_activation)
        if self.output_dropout_rate > 0.0:
            layers.append(nn.Dropout(p=output_dropout_rate))

        # Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if not self.training and not self.dropout_at_inference:
            self.model.eval()
        return self.model(x)
