import torch
import torch.nn as nn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, out_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, user_emb, item_emb):
        batch_size = user_emb.size(0)

        Q = self.query(user_emb).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(item_emb).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(item_emb).view(batch_size, self.num_heads, self.head_dim)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        context = torch.bmm(attention_weights, V)
        context = context.view(batch_size, -1)

        return self.out(context)
