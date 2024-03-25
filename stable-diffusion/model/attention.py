import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            d_embed: int,
            input_projection_bias = True,
            output_projection_bias = True,
        ) -> None:
        super().__init__()

        """
        The Linear layer is commonly used to create a linear transformation from one layer
        to another. It applies a linear transformation to the incoming data. Here's a breakdown
        of its parameters:

        in_features: The size of each input sample. This is the number of features in the input
        data that is fed into the layer. For example, if your input data is a vector of size 10,
        then in_features=10.

        out_features: The size of each output sample. This parameter specifies the size of the
        output from the layer. If you want the layer to output a vector of size 5, you would set
        out_features=5.

        bias: A boolean value that indicates whether a bias vector should be added to the output.
        The default value is True, which means that the layer will learn an additive bias for each
        output feature. If set to False, no bias is added.
        """
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.input_projection = nn.Linear(d_embed, 3 * d_embed, bias=input_projection_bias)

        self.output_projection = nn.Linear(d_embed, d_embed, bias=output_projection_bias)

        self.num_heads = num_heads

        self.d_heads = d_embed // num_heads

    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: (batch_size, seq_length, dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.num_heads, self.d_heads)

        # (batch_size, sequence_length, dimension) -> (batch_size, sequence_length, dimension * 3)
        # chunk => 3 tensors of shape (batch_size, sequence_length, dimension)
        query, key, value = self.input_projection(x).chunk(3, dim=-1)

        # (batch_size, sequence_length, dimension) -> (batch_size, sequence_length, head, dimension / head)
        # transpose => (batch_size, head, sequence_length, dimension / head)
        query = query.view(interim_shape).transpose(1, 2)
        key = key.view(interim_shape).transpose(1, 2)
        value = value.view(interim_shape).transpose(1, 2)

        # (batch_size, head, sequence_length, sequence_length)
        weight = query @ key.transpose(-1, -2)

        if causal_mask:
            # https://pytorch.org/docs/stable/generated/torch.ones_like.html
            # https://pytorch.org/docs/stable/generated/torch.triu.html
            # mask where the upper triangle (above the principal diagonal) is maded up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)

            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, heads, sequence_length, sequence_length) @ (batch_size, heads, sequence_length, dimensions / heads)
        # => (batch_size, heads, sequence_length, dimension / heads)
        output = weight @ value

        # (batch_size, heads, sequence_length, dimension / heads) -> (batch_size, sequence_length, heads, dimension / heads)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.output_projection(output)

        # (batch_size, sequence_length, dimension)
        return output

class CrossAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            d_embed: int,
            d_cross = int,
            input_projection_bias = True,
            output_projection_bias = True,
        ) -> None:
        super().__init__()

        self.query_projection = nn.Linear(d_embed, d_embed, bias=input_projection_bias)
        self.key_projection = nn.Linear(d_cross, d_embed, bias=input_projection_bias)
        self.values_projection = nn.Linear(d_cross, d_embed, bias=input_projection_bias)

        self.output_projection = nn.Linear(d_embed, d_embed, bias=output_projection_bias)

        self.num_heads = num_heads
        self.d_heads = d_embed // num_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x (latent): (batch_size, sequence_length_query, dim_query)
        # y (context): (batch_size, sequence_length_key_value, dim_key_value)

        input_shape = x.shape
        batch_size, sequence_length_query = d_embed = input_shape

        interim_shape = (batch_size, -1, self.num_heads, self.d_heads)

        # multiply query by Wq
        query = self.query_projection(x)
        key = self.key_projection(y)
        value = self.values_projection(y)

        query = query.view(interim_shape).transpose(1, 2)
        key = key.view(interim_shape).transpose(1, 2)
        value = value.view(interim_shape).transpose(1, 2)

        weight = query @ key.transpose(-1, -2)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        output = weight @ value

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.output_projection(output)

        return output