import pytest
import torch
import math
from model import (
    InputEmbedding,
    PositionalEncoding,
    LayerNormalization,
    FeedForwardBlock,
    MultiHeadAttention,
    ResidualConnection,
    EncoderBlock,
    DecoderBlock,
    build_transformer,
)
from dataset import causal_mask
from translate import greedy_decode


def test_input_embedding():
    vocab_size = 100
    d_model = 64
    emb = InputEmbedding(d_model, vocab_size)
    x = torch.tensor([[1, 5, 20], [0, 2, 4]], dtype=torch.int64)
    out = emb(x)
    assert out.shape == (2, 3, d_model)
    assert not torch.isnan(out).any()


def test_positional_encoding():
    d_model = 64
    seq_len = 50
    dropout = 0.0
    pe = PositionalEncoding(d_model, seq_len, dropout)
    x = torch.zeros(2, 20, d_model)
    out = pe(x)
    assert out.shape == (2, 20, d_model)
    # Ensure pe buffer is registered
    assert hasattr(pe, 'pe')
    assert pe.pe.shape == (1, seq_len, d_model)


def test_layer_normalization():
    features = 32
    ln = LayerNormalization(features)
    x = torch.randn(4, 10, features) * 5 + 3
    out = ln(x)
    assert out.shape == (4, 10, features)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1)


def test_feed_forward():
    d_model = 64
    d_ff = 256
    ff = FeedForwardBlock(d_model, d_ff, dropout=0.1)
    x = torch.randn(2, 15, d_model)
    out = ff(x)
    assert out.shape == (2, 15, d_model)


def test_multi_head_attention():
    d_model = 64
    h = 8
    mha = MultiHeadAttention(d_model, h, dropout=0.0)
    q = torch.randn(2, 10, d_model)
    k = torch.randn(2, 10, d_model)
    v = torch.randn(2, 10, d_model)
    mask = None
    out = mha(q, k, v, mask)
    assert out.shape == (2, 10, d_model)


def test_causal_mask():
    mask = causal_mask(5)
    assert mask.shape == (1, 5, 5)
    # The upper triangle (excluding diagonal) must be False
    for i in range(5):
        for j in range(5):
            if j > i:
                assert mask[0, i, j] == False
            else:
                assert mask[0, i, j] == True


def test_encoder_and_decoder_blocks():
    d_model = 64
    h = 4
    d_ff = 128
    enc_block = EncoderBlock(
        d_model,
        MultiHeadAttention(d_model, h, 0.1),
        FeedForwardBlock(d_model, d_ff, 0.1),
        dropout=0.1
    )
    src = torch.randn(2, 10, d_model)
    src_mask = torch.ones(2, 1, 1, 10)
    enc_out = enc_block(src, src_mask)
    assert enc_out.shape == (2, 10, d_model)

    dec_block = DecoderBlock(
        d_model,
        MultiHeadAttention(d_model, h, 0.1),
        MultiHeadAttention(d_model, h, 0.1),
        FeedForwardBlock(d_model, d_ff, 0.1),
        dropout=0.1
    )
    tgt = torch.randn(2, 10, d_model)
    tgt_mask = torch.ones(2, 1, 10, 10)
    dec_out = dec_block(tgt, enc_out, src_mask, tgt_mask)
    assert dec_out.shape == (2, 10, d_model)


def test_full_transformer_forward():
    src_vocab = 50
    tgt_vocab = 60
    seq_len = 16
    d_model = 32
    transformer = build_transformer(
        src_vocab, tgt_vocab, seq_len, seq_len,
        d_model=d_model, n_layers=2, n_heads=4, d_ff=64
    )

    src = torch.randint(0, src_vocab, (2, seq_len))
    tgt = torch.randint(0, tgt_vocab, (2, seq_len))
    src_mask = torch.ones(2, 1, 1, seq_len)
    tgt_mask = causal_mask(seq_len)

    enc = transformer.encode(src, src_mask)
    assert enc.shape == (2, seq_len, d_model)

    dec = transformer.decode(enc, src_mask, tgt, tgt_mask)
    assert dec.shape == (2, seq_len, d_model)

    proj = transformer.project(dec)
    assert proj.shape == (2, seq_len, tgt_vocab)


class DummyTokenizer:
    def __init__(self):
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[SOS]': 2, '[EOS]': 3, 'hello': 4, 'world': 5}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def token_to_id(self, token):
        return self.vocab.get(token, 1)

    def decode(self, ids):
        return " ".join([self.inv_vocab.get(int(i), '[UNK]') for i in ids if int(i) not in (0, 2, 3)])


def test_greedy_decode():
    device = torch.device('cpu')
    tok_src = DummyTokenizer()
    tok_tgt = DummyTokenizer()

    transformer = build_transformer(
        len(tok_src.vocab), len(tok_tgt.vocab), 10, 10,
        d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).to(device)

    src = torch.tensor([[2, 4, 5, 3, 0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    src_mask = (src != 0).unsqueeze(0).unsqueeze(0).int()

    out_tokens = greedy_decode(transformer, src, src_mask, tok_src, tok_tgt, max_len=10, device=device)
    assert out_tokens.dim() == 1
    assert out_tokens.size(0) <= 10
    # First token must be [SOS] (2)
    assert out_tokens[0].item() == 2
