#!/usr/bin/env python
import torch.nn as nn

# Convenience method for specifying common activation functions
def get_activation_function(activation_name):
	"""
	Get activation function module based on its name.

	Args:
		activation_name (str): Name of the activation function.

	Returns:
		torch.nn.Module: Activation function module.
	"""
	activation_function_dict = {
		'sigmoid': nn.Sigmoid(),
		'tanh': nn.Tanh(),
		'relu': nn.ReLU(),
		'leaky_relu': nn.LeakyReLU(negative_slope=0.05), # TODO: Make this a parameter
		'elu': nn.ELU(),
		'selu': nn.SELU(),
		'gelu': nn.GELU(),
		'swish': nn.SiLU(),
		'prelu': nn.PReLU(),
		'softmax': nn.Softmax(dim=-1), # TODO: Make this a parameter
		'identity': nn.Identity(),
		'mish': nn.Mish(),
		'softplus': nn.Softplus(),
		'log_softmax': nn.LogSoftmax(dim=-1), # TODO: Make this a parameter
		'log_sigmoid': nn.LogSigmoid()
	}

	return activation_function_dict.get(activation_name, None)


