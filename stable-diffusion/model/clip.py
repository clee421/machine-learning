import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

# Number of heads for the multi-head attention layer
_NUMBER_OF_HEADS = 12

# Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented
# https://huggingface.co/transformers/v4.8.0/model_doc/clip.html#cliptextconfig
_VOCABULARY_SIZE = 49408

# Vector representation of the words
_NUM_EMBEDDING_VECTORS = 768

# Maximum sequence length that includes the padding
_MAXIMUM_SEQUENCE_LENGTH = 77

_CLIP_LAYER_SIZE = 12

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, num_tokens: int) -> None:
        super().__init__()

        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        # https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        # https://pytorch.org/docs/stable/generated/torch.zeros.html
        self.position_embedding = nn.Parameter(torch.zeros(num_tokens, embedding_size))

    def forward(self, tokens):
        # (batch_size, sequence_length) -> (batch_size, sequence_length, dimension)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, num_heads: int, num_embeds: int) -> None:
        super().__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.layer_norm_1 = nn.LayerNorm(num_embeds)

        self.attention = SelfAttention(num_heads, num_embeds)

        self.layer_norm_2 = nn.LayerNorm(num_embeds)

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear_1 = nn.Linear(num_embeds, 4 * num_embeds)
        self.linear_2 = nn.Linear(4 * num_embeds, num_embeds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, dimension)
        residue = x

        ## Applying Self Attention
        x = self.layer_norm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        ## Feed Forward Layer
        residue = x
        x = self.layer_norm_2(x)
        x = self.linear_1(x)

        # This is the quick GeLU function as opposed to the normal GeLU
        # from https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # For converting the text tokens into numbers
        self.embedding = CLIPEmbedding(_VOCABULARY_SIZE, _NUM_EMBEDDING_VECTORS, _MAXIMUM_SEQUENCE_LENGTH)

        self.layers = nn.Module([
            CLIPLayer(_NUMBER_OF_HEADS, _NUM_EMBEDDING_VECTORS) for i in range(_CLIP_LAYER_SIZE)
        ])

        self.layer_norm = nn.LayerNorm(_NUM_EMBEDDING_VECTORS)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, sequence_length) -> (batch_size, sequence_length, dimension)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (batch_size, sequence_length, dimension)
        output = self.layer_norm(state)

        return output