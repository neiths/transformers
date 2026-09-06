# Building a Transformer from Scratch in Rust: A Beginner's Guide

Welcome to writing your first deep learning model in Rust! If you already understand the Python/PyTorch implementation in this repository, you already know the **math and architecture**. All you need now is learning how those concepts translate into Rust.

---

## 1. Choosing the Right Rust Framework

In Rust, the leading tensor/deep learning framework for Transformers is **[Candle](https://github.com/huggingface/candle)** (created by Hugging Face):
* **Why Candle?**
  * Built specifically for Transformers and LLMs.
  * Minimalist, lightweight, and pure Rust (no heavy C++ LibTorch compilation required).
  * Syntax feels very natural if you are coming from PyTorch (`Tensor`, `Shape`, `matmul`, `transpose`, `softmax`).
  * Seamless support for CPU, CUDA (NVIDIA GPUs), and Metal (Apple Silicon).

---

## 2. PyTorch vs. Rust Mental Model

Coming from Python, here are the key mental shifts:

| PyTorch (Python) | Candle (Rust) | Why? |
| :--- | :--- | :--- |
| `class EncoderBlock(nn.Module):` | `struct EncoderBlock { ... }`<br>`impl EncoderBlock { ... }` | Rust has no classes or inheritance. Data is stored in `struct`, behavior in `impl`. |
| `def forward(self, x):` | `pub fn forward(&self, x: &Tensor) -> Result<Tensor>` | Rust functions explicitly state borrowing (`&`) and return `Result` for error safety. |
| Automatic exceptions (`raise ...`) | The `?` operator (`let y = x.matmul(&w)?;`) | In Rust, operations that could fail (e.g., mismatched tensor dimensions) return `Result<Tensor, Error>`. The `?` unwraps the value or returns the error. |
| `x = x.to(device)` | `Tensor::new(..., &device)?` | You specify device (`Device::Cpu` or `Device::new_cuda(0)?`) at creation. |
| Automatic garbage collection | Strict Ownership & Borrowing | Passing `&Tensor` borrows the tensor without cloning its data. |

---

## 3. Project Setup

### Step 1: Create a New Cargo Project
```bash
cargo new transformer-rust --bin
cd transformer-rust
```

### Step 2: Configure `Cargo.toml`
Open `Cargo.toml` and add Candle dependencies:

```toml
[package]
name = "transformer-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
anyhow = "1.0"

# Enable CUDA if you have an NVIDIA GPU (optional)
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

---

## 4. Step-by-Step Code Walkthrough

We will build the components in `src/model.rs` and wire them up in `src/main.rs`.

---

### Step 1: Positional Encoding
In Rust, we define the `PositionalEncoding` struct storing the precomputed sinusoidal buffer:

```rust
use candle_core::{Device, Result, Tensor};

pub struct PositionalEncoding {
    pe: Tensor,
    dropout: f32,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, dropout: f32, device: &Device) -> Result<Self> {
        let mut pe = vec![0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div_term = (-(2 * i) as f64 * (10000.0f64.ln() / d_model as f64)).exp() as f32;
                let angle = pos as f32 * div_term;
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        let pe = Tensor::from_vec(pe, (1, max_len, d_model), device)?;
        Ok(Self { pe, dropout })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        // Slice the buffer to match current sequence length: pe[:, :seq_len, :]
        let pe_slice = self.pe.narrow(1, 0, seq_len)?;
        x.broadcast_add(&pe_slice)
    }
}
```

---

### Step 2: Layer Normalization
Candle provides `candle_nn::layer_norm`, but here is how a custom LayerNorm is written:

```rust
pub struct LayerNorm {
    alpha: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(features: usize, eps: f64, device: &Device) -> Result<Self> {
        let alpha = Tensor::ones((features,), candle_core::DType::F32, device)?;
        let bias = Tensor::zeros((features,), candle_core::DType::F32, device)?;
        Ok(Self { alpha, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (variance + self.eps)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;
        let scaled = normalized.broadcast_mul(&self.alpha)?;
        scaled.broadcast_add(&self.bias)
    }
}
```

---

### Step 3: Feed-Forward Network ($d_{model} \to d_{ff} \to d_{model}$)
Using Candle's `Linear` layer:

```rust
use candle_nn::{linear, Linear, VarBuilder};

pub struct FeedForwardBlock {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForwardBlock {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = linear(d_ff, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}
```

---

### Step 4: Multi-Head Attention
This implements Scaled Dot-Product Attention: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$

```rust
use candle_nn::ops::softmax;

pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    h: usize,
    d_k: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, h: usize, vb: VarBuilder) -> Result<Self> {
        assert_eq!(d_model % h, 0, "d_model must be divisible by h");
        let d_k = d_model / h;
        let w_q = linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = linear(d_model, d_model, vb.pp("w_k"))?;
        let w_v = linear(d_model, d_model, vb.pp("w_v"))?;
        let w_o = linear(d_model, d_model, vb.pp("w_o"))?;
        Ok(Self { w_q, w_k, w_v, w_o, h, d_k })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len_q, _) = q.dims3()?;
        let (_, seq_len_k, _) = k.dims3()?;

        // Project Q, K, V
        let query = self.w_q.forward(q)?;
        let key = self.w_k.forward(k)?;
        let value = self.w_v.forward(v)?;

        // Reshape: (B, S, d_model) -> (B, S, h, d_k) -> (B, h, S, d_k)
        let query = query.reshape((b_sz, seq_len_q, self.h, self.d_k))?.transpose(1, 2)?.contiguous()?;
        let key = key.reshape((b_sz, seq_len_k, self.h, self.d_k))?.transpose(1, 2)?.contiguous()?;
        let value = value.reshape((b_sz, seq_len_k, self.h, self.d_k))?.transpose(1, 2)?.contiguous()?;

        // Attention scores: (Q @ K.T) / sqrt(d_k)
        let scale = 1.0 / (self.d_k as f64).sqrt();
        let key_t = key.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?.contiguous()?;
        let mut scores = (query.matmul(&key_t)? * scale)?;

        // Apply mask if provided
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }

        // Softmax along last dimension
        let weights = softmax(&scores, candle_core::D::Minus1)?;

        // Weighted values: Weights @ V -> (B, h, S_q, d_k)
        let context = weights.matmul(&value)?;

        // Concatenate heads: (B, h, S, d_k) -> (B, S, h, d_k) -> (B, S, d_model)
        let context = context.transpose(1, 2)?.contiguous()?.reshape((b_sz, seq_len_q, self.h * self.d_k))?;

        // Output projection
        self.w_o.forward(&context)
    }
}
```

---

### Step 5: Encoder Block & Encoder
Stacking layers with Pre-LayerNorm:

```rust
pub struct EncoderBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForwardBlock,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderBlock {
    pub fn new(d_model: usize, h: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(d_model, h, vb.pp("self_attention"))?;
        let feed_forward = FeedForwardBlock::new(d_model, d_ff, vb.pp("feed_forward"))?;
        let norm1 = LayerNorm::new(d_model, 1e-6, vb.device())?;
        let norm2 = LayerNorm::new(d_model, 1e-6, vb.device())?;
        Ok(Self { attention, feed_forward, norm1, norm2 })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-LN Self-Attention
        let norm_x = self.norm1.forward(x)?;
        let attn_out = self.attention.forward(&norm_x, &norm_x, &norm_x, mask)?;
        let x = (x + attn_out)?;

        // Pre-LN Feed-Forward
        let norm_x = self.norm2.forward(&x)?;
        let ff_out = self.feed_forward.forward(&norm_x)?;
        x + ff_out
    }
}
```

---

### Step 6: Causal Mask for Autoregressive Decoding
Creating the upper-triangular mask where future tokens are filled with `-f32::INFINITY`:

```rust
pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = -1e9; // Large negative value for -infinity
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}
```

---

## 5. Running Your First Model in `src/main.rs`

Here is an end-to-end executable test in `src/main.rs`:

```rust
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

mod model;
use model::{causal_mask, EncoderBlock};

fn main() -> Result<()> {
    // 1. Choose Device (CUDA GPU or CPU)
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Running on device: {:?}", device);

    // 2. Dummy weights builder (Zeros for initialization demo)
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

    // 3. Instantiate Encoder Block: d_model = 64, heads = 4, d_ff = 256
    let block = EncoderBlock::new(64, 4, 256, vb)?;

    // 4. Create dummy input tensor: Batch = 2, Seq_len = 10, d_model = 64
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device)?;

    // 5. Forward pass
    let output = block.forward(&input, None)?;

    println!("Input shape:  {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    assert_eq!(output.dims(), &[2, 10, 64]);

    println!("\n Transformer Encoder forward pass in Rust successful!");
    Ok(())
}
```

Run it with:
```bash
cargo run
```

---

## 6. How to Load Your PyTorch Weights (`tmodel_00.pt`) into Rust

You don't have to train from scratch in Rust! You can convert your PyTorch checkpoint to **Safetensors** (the universal standard) and load it directly into Rust:

### Step 1: Export in Python (`export_safetensors.py`)
```python
import torch
from safetensors.torch import save_file

checkpoint = torch.load("weights/tmodel_00.pt", map_location="cpu")
save_file(checkpoint["model_state_dict"], "weights/model.safetensors")
print("Exported to weights/model.safetensors!")
```

### Step 2: Load in Rust
```rust
use candle_nn::VarBuilder;

let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["weights/model.safetensors"], candle_core::DType::F32, &device)?
};
```
Candle maps the weights directly from disk into GPU memory without extra copies!

---

## 7. Recommended Learning Roadmap for a Rust Beginner

1. **Week 1: Core Rust Fundamentals**
   * Read the official free book: *[The Rust Programming Language](https://doc.rust-lang.org/book/)* (Chapters 1 to 10 cover ownership, borrowing, structs, and results).
   * Solve exercises on your machine using `rustlings` (already installed in your terminal!).
2. **Week 2: Candle Tensor Basics**
   * Experiment with Candle's `Tensor` operations: reshaping, slicing (`narrow`), matrix multiplication, and broadcasting.
3. **Week 3: Encoder-Decoder Architecture**
   * Implement the code in this guide and verify tensor shapes against your Python implementation.
4. **Week 4: Safetensors & Tokenizer**
   * Use the `tokenizers` crate in Rust (`tokenizers::Tokenizer::from_file("tokenizer_en.json")`) to connect input text to your Rust Transformer!
