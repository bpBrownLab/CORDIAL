#!/usr/bin/env python

# General imports
from argparse import Namespace
from typing import Union

# PyTorch imports
from torch.nn import Module

# Project imports
from modules.architectures.model_factory import ModelFactory
# from modules.datasets.generic_dataset import GenericDataset
from modules.datasets.interaction_graph_dataset_legacy import InteractionGraphDatasetLegacy

class ModelInitializer:
    """Initialize machine learning models with dataset-specific parameters."""
    def __init__(self, args: Namespace, dataset: Union[InteractionGraphDatasetLegacy], model_str: str = 'cordial'):
        """
        Initialize model with configuration and dataset.

        Args:
            args: Configuration arguments
            dataset: Dataset for model initialization
            model_str: Model type ('cordial')
        """
        self.args = args
        self.dataset = dataset
        self.model_str = model_str
        
        if self.args is None:
            raise ValueError("Cannot initialize a model without arguments!")

        # Initialize
        self.model = self.init()
        
        if self.model is None:
            raise ValueError(f"Failed to initialize model. Check your 'model_type' string: {self.model_str}")

    def init(self) -> Module:
        """Initialize the specified model."""
        if self.model_str == 'cordial':
            return self.init_cordial()

    def init_cordial(self) -> Module:
        """Initialize CORDIAL model with dataset-specific parameters."""
        if isinstance(self.dataset, InteractionGraphDatasetLegacy):
            print("Initializing model from InteractionGraphDatasetLegacy.")
            
            return ModelFactory.create_model(
                self.model_str,
                num_feature_columns=self.dataset.num_feature_columns(),
                num_distance_bins=self.dataset.num_distance_bins,
                kernel_size=self.args.kernel_size,
                conv_channels=self.args.conv_channels,
                hidden_size=self.args.hidden_layers,
                fc_dropout=self.args.fc_dropout,
                conv_dropout=self.args.conv_dropout,
                activation_function_names=self.args.activation_function_names,
                output_size=self.args.num_result_classes,
                num_attn_heads=self.args.attention_heads,
                num_row_attn_heads=self.args.num_row_attn_heads,
                num_column_attn_heads=self.args.num_column_attn_heads,
                num_attn_layers=self.args.num_attn_layers,
                attention_dropout=self.args.attention_dropout,
                debug=self.args.debug,
                disable_conv1d=self.args.disable_conv1d,
                disable_attention=self.args.disable_attention
            )
        
        else:
            print("Failed to initialize CORDIAL!")
