#!/usr/bin/env python

# Standard imports
from typing import Dict, Tuple, Optional, List
import math
import time

# PyTorch imports
import torch
import torch.nn as nn

# Project imports
from modules.architectures.mlp import MLP
from modules.architectures.axial_attention_with_norm import AxialAttentionWithNorm

# Type aliases for clarity
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]
OptionalTensor = Optional[Tensor]

"""
CORDIAL: COnvolutional Representation of Distance-dependent Interactions with Attention Learning.

A neural network architecture for molecular interaction prediction using distance-dependent features.
Processes chemical interaction radial distribution functions through convolutional layers and 
axial attention mechanisms.

Architecture: Input --> Conv1D --> Axial Attention --> MLP --> Output

The model learns both local distance patterns and global correlations for binding affinity prediction.
"""

class CORDIAL(nn.Module):
    """
    CORDIAL neural network for molecular interaction prediction.
    
    Processes distance-feature matrices through convolutional layers for local pattern 
    recognition and axial attention for global context modeling.
    
    Components:
        - Conv1D layers: Feature-specific distance pattern learning
        - Axial attention: Global distance-feature correlations  
        - MLP: Final multiclass ordinal classification
    """
    def __init__(self,
                 num_feature_columns, num_distance_bins, hidden_size, output_size=8,
                 kernel_size=7, conv_channels=4, conv_dropout=0.05,
                 attention_dropout=0.15, 
                 num_attn_heads=1,
                 num_row_attn_heads=1,
                 num_column_attn_heads=1,
                 num_attn_layers=1,
                 fc_dropout=0.25, 
                 activation_function_names='mish',
                 input_dropout=0.0,
                 debug=False,
                 disable_conv1d=False,
                 disable_attention=False):
        """
        Initialize the CORDIAL model.

        Args:
            num_feature_columns (int): Number of feature columns in the input data.
            num_distance_bins (int): Number of distance bins for the interaction data.
            hidden_size (list): Dimensions of the hidden layers in the feed-forward network.
            output_size (int): Dimension of the output layer.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 7.
            conv_channels (int, optional): Number of convolutional channels. Defaults to 4.
            conv_dropout (float, optional): Dropout rate for convolutional layers. Defaults to 0.05.
            fc_dropout (float or list, optional): Dropout rate(s) for the model. Defaults to 0.25.
            activation_function_names (list, optional): Names of activation functions for hidden layers.
            num_attn_heads (int, optional): Number of attention heads. Defaults to 1.
            num_row_attn_heads (int, optional): Number of row attention heads. Defaults to 1.
            num_column_attn_heads (int, optional): Number of column attention heads. Defaults to 1.
            num_attn_layers (int, optional): Number of attention layers. Defaults to 1.
            attention_dropout (float, optional): Dropout rate for attention layers. Defaults to 0.15.
            debug (bool, optional): Enable debug mode. Defaults to False.
            disable_conv1d (bool, optional): Disable 1D convolutions. Defaults to False.
            disable_attention (bool, optional): Disable inner transformer. Defaults to False.
            input_dropout (float, optional): Dropout rate for input features. Defaults to 0.0.
        """
        super(CORDIAL, self).__init__()
        
        # Feature parameters
        self.num_feature_columns = num_feature_columns
        self.num_distance_bins = num_distance_bins
        
        # Convolutional parameters
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels if isinstance(conv_channels, list) else [conv_channels]
        self.conv_dropout = conv_dropout

        # Attention parameters
        self.num_attn_heads = num_attn_heads  # For standard attention
        self.num_row_attn_heads = num_row_attn_heads  # For axial attention
        self.num_column_attn_heads = num_column_attn_heads  # For axial attention
        self.num_attn_layers = num_attn_layers
        self.attention_dropout = attention_dropout
        
        # MLP parameters
        self.hidden_size = hidden_size
        self.fc_dropout = fc_dropout
        self.activation_function_names = activation_function_names
        self.output_size = output_size

        # General parameters
        self.debug = debug
        self.disable_conv1d = disable_conv1d
        self.disable_attention = disable_attention

        #############################
        #      Convolutions         #
        #############################
        """
        Convolutional block for learning distance-dependent features

        My overarching goal with the convolutional layers was just to learn feature-specific patterns as a function of distance that would then be used to learn global relationships through the attention mechanism. Effectively, we are learning appropriate smoothing functions for each feature.
        """
        # Combined convolution block for learning distance-dependent features
        self.conv = nn.Sequential(
            # Feature-specific pattern learning with grouped convolution
            nn.Conv1d(in_channels=self.num_feature_columns, 
                     out_channels=self.num_feature_columns*self.conv_channels[0], 
                     kernel_size=7, 
                     padding='same', 
                     padding_mode='replicate', 
                     stride=1, 
                     bias=False,
                     groups=self.num_feature_columns),
            
            nn.BatchNorm1d(self.num_feature_columns*self.conv_channels[0]),
            nn.GELU(),
            nn.Dropout1d(p=self.conv_dropout),

            # Channel reduction with local smoothing
            nn.Conv1d(in_channels=self.num_feature_columns*self.conv_channels[0], 
                     out_channels=self.num_feature_columns, 
                     kernel_size=3, 
                     padding='same', 
                     padding_mode='replicate',
                     stride=1,
                     bias=True,
                     groups=self.num_feature_columns)
        )

        # Initialize the convolutional blocks
        self._initialize_convblock()

        #############################
        #         Attention         #
        #############################
        """
        Self-attention captures long-range dependencies between convolved distance bins that convolutions miss. I figured that this would better allow learning correlations between any pair of distance bins (e.g., H-bond at 2.5 angstroms influencing 6.0 angstroms interactions) and provide global context for understanding overall interaction profiles
        
        """
        
        self.self_attention = nn.ModuleList([
            AxialAttentionWithNorm(
                embed_dim=self.num_feature_columns,
                num_row_heads=self.num_row_attn_heads,
                num_column_heads=self.num_column_attn_heads,
                dropout=self.attention_dropout,
                use_row_positional_encoding=True,  # Distance bins have meaningful order
                use_column_positional_encoding=False,
                norm_first=True,
                use_residual=True,
                use_feed_forward=True,
                ff_expansion=4
            ) for _ in range(self.num_attn_layers)
        ])
        
        #############################
        #    Multiayer Perceptron   #
        #############################

        # Feed-forward network for final prediction
        if len(self.activation_function_names) != len(self.hidden_size):
            print(f"WARNING: Number of hidden layers ({len(self.hidden_size)}) not equal to number of activation functions "
                  f"({len(self.activation_function_names)}).")
            print("WARNING: Defaulting to setting 'mish' activation functions for each hidden layer!")
            self.activation_function_names = ['mish'] * len(self.hidden_size)

        if len(self.fc_dropout) != len(self.hidden_size) + 1:
            print(f"WARNING: Number of dropout rates ({len(self.fc_dropout)}) not equal to combined number of input and hidden layers ("
                  f"{len(self.hidden_size) + 1}).")
            print("WARNING: Defaulting to setting 'dropout_rates' to 10% for all layers!")
            self.fc_dropout = [0.10] * (len(self.hidden_size) + 1)

        # Fully connected layer(s)
        self.fc = MLP(
            input_size=self.num_distance_bins * self.num_feature_columns,
            hidden_sizes=self.hidden_size,
            output_size=self.output_size,
            activation_function_names=self.activation_function_names,
            output_activation_function_name=None,
            dropout_rates=self.fc_dropout,
            dropout_at_inference=False
        )

        # Hooks for visualization
        self.hooks = []
        self.activation = {}

        # Print model statistics
        param_counts = self.get_parameter_count()
        print("\nCORDIAL Model Statistics:")
        print(f"Convolutional layers: {param_counts['conv']:,} parameters")
        print(f"Attention mechanism: {param_counts['attention']:,} parameters")
        print(f"MLP layers: {param_counts['mlp']:,} parameters")
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Estimated model size: {param_counts['total'] * 4 / (1024*1024):.2f} MB\n")

        # Add input dropout layer
        self.input_dropout = nn.Dropout1d(p=input_dropout) if input_dropout > 0 else None

    def forward(self, batch):
        """
        Forward pass through the CORDIAL network.

        Args:
            batch (dict): Batch containing 'features' tensor of shape 
                         [batch_size, num_distance_bins, num_feature_columns]

        Returns:
            tuple: (predictions, pre_conv_features, conv_output, attn_output, attention_weights, activations)
        """
        forward_start_time = time.time()
        # Clear activations dict if not needed
        if not self.debug:
            self.activation.clear()

        x = batch['features']
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._debug_print(f'x: {x.size()}')
        assert not torch.isnan(x).any(), "Input contains NaNs!"
        batch_size = x.size()[0]
        self._debug_print(f'batch_size: {batch_size}')
        self._debug_print(f'x min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}')

        # Apply input dropout if enabled
        if self.input_dropout is not None:
            # Permute to [batch_size, feature_columns, distance_bins] for Dropout1d
            x = x.permute(0, 2, 1)
            x = self.input_dropout(x)
            # Permute back to [batch_size, distance_bins, feature_columns]
            x = x.permute(0, 2, 1)
            self._debug_print(f'x after input dropout: {x.size()}')
            self._debug_print(f'x min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}')
            assert not torch.isnan(x).any(), "Input dropout output contains NaNs!"

        # This is the feature map that will be used as input to the main blocks.
        x_pre_conv = x.clone()

        # Apply 1D convolutions across the distances bins separately for each feature column.
        conv_start_time = time.time()
        if not self.disable_conv1d:
            x = self._apply_convolutions(x)
            conv_output = x.clone()
            self._debug_print(f'x after convolutions: {x.size()}')
            self._debug_print(f'x min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}')
            assert not torch.isnan(x).any(), "Convolution output contains NaNs!"
        else:
            conv_output = None
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        conv_time = time.time() - conv_start_time

        # Apply self-attention / transformer across distance bin sequence space
        attention_start_time = time.time()
        if not self.disable_attention:
            x, attention_weights = self._apply_attention(x)
            if conv_output is not None:
                x += conv_output
            attn_output = x.clone()
            self._debug_print(f'x after attention: {x.size()}')
            self._debug_print(f'x min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}')
            assert not torch.isnan(x).any(), "Inner transformer output contains NaNs!"
        else:
            attn_output = None
            attention_weights = None
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        attention_time = time.time() - attention_start_time

        # Apply MLP for final prediction
        mlp_start_time = time.time()
        x = x.reshape(batch_size, -1)
        x = self._apply_mlp(x)
        assert not torch.isnan(x).any(), "Fully connected output contains NaNs!"
        self._debug_print(f'output min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}')
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mlp_time = time.time() - mlp_start_time
        
        total_forward_time = time.time() - forward_start_time
        if self.debug:
            print("[TIMER] CORDIAL forward pass breakdown:")
            print(f"  - Convolutions: {conv_time:.6f}s")
            print(f"  - Attention: {attention_time:.6f}s")
            print(f"  - MLP: {mlp_time:.6f}s")
            print(f"  - Total: {total_forward_time:.6f}s")

        # Return the output, auxiliary loss data, attention weights, and activations
        return x, x_pre_conv, conv_output, attn_output, attention_weights, self.activation

    def _debug_print(self, message):
        """
        Prints a debug message if the debug flag is set to True.

        Parameters:
            message (str): The message to print.
        """
        if self.debug:
            print(message)

    def _initialize_convblock(self):
        for layer in [self.conv]:
            if isinstance(layer, nn.Conv1d):
                # Calculate gain for GELU (approximately 1.7 for small inputs)
                gelu_gain = math.sqrt(2.0 / (1 + 0.666))  # ~1.22
                
                # Use fan_out for grouped convs, scale for GELU
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode='fan_out',
                    nonlinearity='linear'  # Manual scaling
                )
                layer.weight.data *= gelu_gain
                
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def _apply_convolutions(self, x: Tensor) -> Tensor:
        """
        Apply 1D convolutions across distance bins.

        Args:
            x: Input tensor [batch_size, distance_bins, feature_columns]

        Returns:
            Convolved tensor with same shape as input
        """
        self._debug_print(f'[batch_size, distance_bins, feature_columns] before convolution: {x.size()}')
        x = x.permute(0, 2, 1) # Permute to [batch_size, feature_columns, distance_bins] for Conv1d
        x = self.conv(x)
        x = x.permute(0, 2, 1) # Permute back to [batch_size, distance_bins, feature_columns]
        self._debug_print(f'[batch_size, distance_bins, feature_columns] after convolution: {x.size()}')
        return x

    def _apply_attention(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Apply axial attention across distance-feature matrix.

        Args:
            x: Input tensor [batch_size, num_distance_bins, num_feature_columns]

        Returns:
            Processed tensor and list of attention weights
        """
        attention_weights = []
        for i, self_attn in enumerate(self.self_attention):
            x, weights = self_attn(x)
            self._debug_print(f'attention_layer_{i}_output: {x.size()}')
            attention_weights.append((f'self_attention_{i}', weights))
        
        return x, attention_weights

    def _apply_mlp(self, x: Tensor) -> Tensor:
        """
        Apply MLP for final prediction.
        
        Args:
            x: Flattened tensor [batch_size, num_distance_bins * num_feature_columns]
        
        Returns:
            Output predictions
        """
        return self.fc(x)

    def _get_activation(self, name):
        def hook(_, __, output):
            self.activation[name] = output.detach()
        return hook

    def attach_hooks(self):
        for i, conv in enumerate(self.conv):
            self.hooks.append(conv.register_forward_hook(self._get_activation(f'conv_{i}')))
        
        for i, attn_layer in enumerate(self.self_attention):
            self.hooks.append(attn_layer.register_forward_hook(self._get_activation(f'self_attention_{i}')))
        
        # Attach hooks to linear layers in the MLP
        for i, layer in enumerate(self.fc.model):
            if isinstance(layer, nn.Linear):
                self.hooks.append(layer.register_forward_hook(self._get_activation(f'fc_{i}')))

    def detach_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each model component."""
        return {
            'conv': sum(p.numel() for p in self.conv.parameters()),
            'attention': sum(p.numel() for p in self.self_attention.parameters()),
            'mlp': sum(p.numel() for p in self.fc.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
