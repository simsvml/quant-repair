import torch
from torch import nn
from torch.nn import functional as F
from torchtune.modules.quantized import make_quantized_tensor


class WithAdapter(nn.Module):
    def __init__(self, base: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, x):
        return self.base(x) + self.adapter(x)


class LowRankAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank, *, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_a = nn.Parameter(torch.empty((rank, in_features), device=device))
        self.lora_b = nn.Parameter(torch.empty((out_features, rank), device=device))

    def forward(self, x):
        x = F.linear(x, self.lora_a)
        x = F.linear(x, self.lora_b)
        return x

class QuantLowRankAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank, *, lora_quant, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_a = make_quantized_tensor((rank, in_features), lora_quant, device=device)
        self.lora_b = make_quantized_tensor((out_features, rank), lora_quant, device=device)

    def forward(self, x):
        x = F.linear(x, self.lora_a.forward())
        x = F.linear(x, self.lora_b.forward())
        return x

class EmbeddingLowRankAdapter(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank, *, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.lora_a = nn.Parameter(torch.empty((rank, num_embeddings), device=device))
        self.lora_b = nn.Parameter(torch.empty((embedding_dim, rank), device=device))

    def forward(self, x):
        x = F.embedding(x, self.lora_a)
        x = F.linear(x, self.lora_b)
        return x

class QuantEmbeddingLowRankAdapter(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank, *, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.lora_a = make_quantized_tensor((rank, num_embeddings), lora_quant, device=device)
        self.lora_b = make_quantized_tensor((embedding_dim, rank), lora_quant, device=device)

    def forward(self, x):
        x = F.embedding(x, self.lora_a.forward())
        x = F.linear(x, self.lora_b.forward())
        return x
