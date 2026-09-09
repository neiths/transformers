use candle_core::{Device, Result, Tensor};

pub struct PositionalEncoding {
    pe: Tensor,
    dropout: f32,
}
