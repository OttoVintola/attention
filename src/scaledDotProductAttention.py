import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class ScaledDotProductAttention():
    def __init__(self, dropout: float = 0.1, temperature: float = 1.0):
        """
        Initialize the Scaled Dot-Product Attention module.
        Args:
            dropout (float): Dropout rate.
            temperature (float): Temperature scaling factor for the attention scores.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        Compute the attention weights and output.

        Args:
            query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, d_k).
            key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, d_k).
            value (Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, d_v).
            mask (Tensor): Optional mask tensor of shape (batch_size, 1, 1, seq_len_k).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_heads, seq_len_q, d_v).
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output
    
