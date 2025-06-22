```bash
transformer = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=50,
    tgt_seq_len=50,
    d_model=512,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
    d_ff=2048
)
print(transformer)


Transformer(
  (encoder): Encoder(
    (layers): ModuleList(
      (0-5): 6 x EncoderBlock(
        (self_attention_block): MultiHeadAttention(
          (dropout): Dropout(p=0.1, inplace=False)
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_o): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward_block): FeedForwardBlock(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (residual_connections): ModuleList(
          (0-1): 2 x ResidualConnection(
            (dropout): Dropout(p=0.1, inplace=False)
            (norm): LayerNormalization()
          )
        )
      )
    )
    (norm): LayerNormalization()
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0-5): 6 x DecoderBlock(
        (self_attention_block): MultiHeadAttention(
          (dropout): Dropout(p=0.1, inplace=False)
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_o): Linear(in_features=512, out_features=512, bias=True)
        )
        (cross_attention_block): MultiHeadAttention(
          (dropout): Dropout(p=0.1, inplace=False)
          (w_q): Linear(in_features=512, out_features=512, bias=True)
          (w_k): Linear(in_features=512, out_features=512, bias=True)
          (w_v): Linear(in_features=512, out_features=512, bias=True)
          (w_o): Linear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward_block): FeedForwardBlock(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (residual_connections): ModuleList(
          (0-2): 3 x ResidualConnection(
            (dropout): Dropout(p=0.1, inplace=False)
            (norm): LayerNormalization()
          )
        )
      )
    )
    (norm): LayerNormalization()
  )
  (src_embed): InputEmbedding(
    (embedding): Embedding(10000, 512)
  )
  (tgt_embed): InputEmbedding(
    (embedding): Embedding(10000, 512)
  )
  (src_pos): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (tgt_pos): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (projection_layer): ProjectionLayer(
    (proj): Linear(in_features=512, out_features=10000, bias=True)
  )
)
```