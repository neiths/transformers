from torch import nn
import torch
import math


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # exp(ln()) ~ nothing here, minus stands for not using fraction form.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormlization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


if __name__ == "__main__":
    d_model = 512
    vocab_size = 10000
    seq_len = 100

    input_embedding = InputEmbedding(d_model, vocab_size)
    positional_encoding = PositionalEncoding(d_model, seq_len, dropout=0.1)
    layer_norm = LayerNormlization()

    # Example usage
    x = torch.randint(0, vocab_size, (10, seq_len))  # Batch of 10 sequences
    embedded_x = input_embedding(x)
    encoded_x = positional_encoding(embedded_x)

    print(encoded_x.shape)  # Should be (10, seq_len, d_model)

    normalized_x = layer_norm(encoded_x)

    print(encoded_x[0])

    print(normalized_x[0])


