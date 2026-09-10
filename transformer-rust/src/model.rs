use candle_core::{Device, Result, Tensor};

pub struct PositionalEncoding {
    pe: Tensor,
    dropout: f32,
}

impl PositionalEncoding {
    pub fn mew(d_model: usize, max_len: usize, dropout: f32, device: &Device) -> Result<Self> {
        let mut pe = vec![0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div_term = (-(2 * i) as f64 * (10_000.0f64.ln() / d_model as f64)).exp() as f32;

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
