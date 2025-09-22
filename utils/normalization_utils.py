#!/usr/bin/env python

import pickle
import torch


def save_mean_std(mean_tensor, std_tensor, filename):
	"""
	Save mean and standard deviation tensors to a file using pickle.

	Parameters:
	- mean_tensor (torch.Tensor): Tensor of mean values.
	- std_tensor (torch.Tensor): Tensor of standard deviation values.
	- filename (str): Name of the file to save the data.
	"""
	if not isinstance(mean_tensor, torch.Tensor) or not isinstance(std_tensor, torch.Tensor):
		raise TypeError("Input values must be PyTorch tensors.")

	data = {'mean': mean_tensor, 'std': std_tensor}
	with open(filename, 'wb') as file:
		pickle.dump(data, file)


def load_mean_std(filename):
	"""
	Load mean and standard deviation tensors from a file using pickle.

	Parameters:
	- filename (str): Name of the file containing the data.

	Returns:
	- mean_tensor (torch.Tensor): Tensor of mean values.
	- std_tensor (torch.Tensor): Tensor of standard deviation values.
	"""
	with open(filename, 'rb') as file:
		data = pickle.load(file)

	if not isinstance(data['mean'], torch.Tensor) or not isinstance(data['std'], torch.Tensor):
		raise TypeError("Loaded data must be PyTorch tensors.")

	return data['mean'], data['std']
