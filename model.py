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
        

if __name__ == "__main__":
    
    d_model = 3
    
    vocab_size = 10
    
    input_embedding = InputEmbedding(d_model, vocab_size)
    
    x = torch.LongTensor([1, 2, 3, 4, 5])
    
    output = input_embedding(x)
    
    print(output) 
    