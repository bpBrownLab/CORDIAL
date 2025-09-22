#!/usr/bin/env python
import torch.nn as nn
from modules.architectures.sinusoidal_positional_encoding import SinusoidalPositionalEncoding

class AxialAttentionWithNorm(nn.Module):
    """
    Axial attention module for distance-feature matrices.
    
    Applies attention separately along distance (row) and feature (column) dimensions,
    reducing computational complexity from O(nm) to O(n) + O(m).

    NOTE - contains a known bug where dimension mismatch occurs if numbers of rows and columns are not the same.
    TODO: fix this bug.

    """
    
    def __init__(self, embed_dim, 
                 num_row_heads, 
                 num_column_heads, 
                 dropout=0.15,
                 use_row_positional_encoding=True,  # Distance bins are ordinal
                 use_column_positional_encoding=False,  # Feature order does not contain meaningful information
                 norm_first=True, 
                 use_residual=True, 
                 use_feed_forward=True, 
                 ff_expansion=4):
        super(AxialAttentionWithNorm, self).__init__()
        
        # Row-wise attention (distance dimension)
        self.row_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_row_heads,
            batch_first=True,
            dropout=dropout
        )

        # Column-wise attention (feature dimension)
        self.column_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_column_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Layer normalization
        self.row_norm = nn.LayerNorm(embed_dim)
        self.column_norm = nn.LayerNorm(embed_dim)

        # Positional encoding
        self.use_row_positional_encoding = use_row_positional_encoding
        self.use_column_positional_encoding = use_column_positional_encoding
        
        if self.use_row_positional_encoding:
            self.row_pos_enc = SinusoidalPositionalEncoding()
        if self.use_column_positional_encoding:
            self.column_pos_enc = SinusoidalPositionalEncoding()
        
        self.norm_first = norm_first
        self.use_residual = use_residual
        
        self.use_feed_forward = use_feed_forward
        if use_feed_forward:
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ff_expansion),
                nn.GELU(),
                nn.Linear(embed_dim * ff_expansion, embed_dim)
            )
            self.ff_norm = nn.LayerNorm(embed_dim)

    def _apply_attention(self, q, k, v, attention, norm, pos_enc=None, attention_mask=None):
        """Apply attention with normalization and residual connection."""
        if (self.use_row_positional_encoding and pos_enc is not None) or \
           (self.use_column_positional_encoding and pos_enc is not None):
            q = pos_enc(q)
            k = pos_enc(k)
            v = pos_enc(v)

        if self.norm_first:
            # Pre-norm
            attn_output, attn_weights = attention(
                norm(q), norm(k), norm(v),
                attn_mask=attention_mask
            )
            if self.use_residual:
                attn_output = q + attn_output
        else:
            # Post-norm
            attn_output, attn_weights = attention(q, k, v, attn_mask=attention_mask)
            if self.use_residual:
                attn_output = norm(q + attn_output)
            else:
                attn_output = norm(attn_output)
                
        return attn_output, attn_weights

    def forward(self, q, k=None, v=None, attention_mask=None):
        """
        Apply axial attention to input tensor.

        Args:
            q: Query tensor [batch_size, num_distance_bins, num_features]
            k: Key tensor (defaults to q)
            v: Value tensor (defaults to q)
            attention_mask: Not supported for axial attention

        Returns:
            tuple: (output tensor, (row_weights, column_weights))
        """

        # Terminate with error if attention mask is provided
        if attention_mask is not None:
            raise ValueError("Attention mask is not supported for axial attention.")
        
        # If k and v are not provided, set them to q
        if k is None or v is None:
            k = q
            v = q
        
        # First attend along row dimension
        row_output, row_weights = self._apply_attention(
            q, k, v,
            self.row_attention,
            self.row_norm,
            self.row_pos_enc if self.use_row_positional_encoding else None
        )

        # Then attend along column dimension
        column_q = row_output.transpose(-2, -1)
        column_k = k.transpose(-2, -1) if k is not None else column_q
        column_v = v.transpose(-2, -1) if v is not None else column_q

        column_output, column_weights = self._apply_attention(
            column_q, column_k, column_v,
            self.column_attention,
            self.column_norm,
            self.column_pos_enc if self.use_column_positional_encoding else None
        )
        output = column_output.transpose(-2, -1)
        
        # Feed-forward block (if enabled)
        if self.use_feed_forward:
            if self.norm_first:
                ff_output = self.feed_forward(self.ff_norm(output))
                if self.use_residual:
                    ff_output = output + ff_output
            else:
                ff_output = self.feed_forward(output)
                if self.use_residual:
                    ff_output = self.ff_norm(output + ff_output)
                else:
                    ff_output = self.ff_norm(ff_output)
            
            return ff_output, (row_weights, column_weights)
        
        return output, (row_weights, column_weights)