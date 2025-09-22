import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
	"""
	Implements the sinusoidal positional encoding for transformer models.

	The positional encodings use sine and cosine functions of different frequencies.
	Each dimension of the positional encoding corresponds to a sinusoid.

	"""

	def __init__(self):
		super(SinusoidalPositionalEncoding, self).__init__()

	def forward(self, x):
		"""
		Add sinusoidal positional encodings to input tensor.

		Args:
			x (Tensor): Tensor of shape (batch_size, seq_length, embed_dim)

		Returns:
			Tensor: Tensor with positional encodings added, ensuring same device as input.
		"""

		device = x.device
		position = torch.arange(0, x.size(1), dtype=torch.float, device=device).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, x.size(2), 2, device=device).float() * (-math.log(10000.0) / x.size(2)))

		encoding = torch.zeros(x.size(1), x.size(2), device=device)
		encoding[:, 0::2] = torch.sin(position * div_term[:encoding.size(1) // 2 + encoding.size(1) % 2])
		encoding[:, 1::2] = torch.cos(position * div_term[:encoding.size(1) // 2])

		encoding = encoding.unsqueeze(0).expand(x.size(0), -1, -1)
		return x + encoding.detach()
