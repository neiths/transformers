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


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        #  (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.attention_scores = None
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, Seq_len, Seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores  # (Batch, h, Seq_len, d_k)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        key = self.w_k(k)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        value = self.w_v(v)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        #  (Batch, h, Seq_len, d_k) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)

        return self.w_o(x)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)


if __name__ == "__main__":
    d_model = 512
    vocab_size = 10000
    seq_len = 100

    input_embedding = InputEmbedding(d_model, vocab_size)
    positional_encoding = PositionalEncoding(d_model, seq_len, dropout=0.1)
    layer_norm = LayerNormlization()
    ff_block = FeedForwardBlock(d_model, d_ff=2048, dropout=0.1)
    multi_head_attention = MultiHeadAttention(d_model, h=8, dropout=0.1)

    #  Example usage
    x = torch.randint(0, vocab_size, (10, seq_len))  # Batch of 10 sequences
    embedded_x = input_embedding(x)
    encoded_x = positional_encoding(embedded_x)

    # print(encoded_x.shape)  # Should be (10, seq_len, d_model)

    normalized_x = layer_norm(encoded_x)

    # print(ff_block(normalized_x))

    print(multi_head_attention(normalized_x, normalized_x, normalized_x, None))  # Should be (10, seq_len, d_model)
