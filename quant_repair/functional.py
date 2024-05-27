from dataclasses import dataclass
from typing import Any, Optional, List
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from quant_repair import quantized


@dataclass(frozen=True)
class TensorParams:
    data: Tensor
    dequant: quantized.DequantizeParams

    @classmethod
    def from_unquantized_tensor(cls, x: Tensor) -> 'TensorParams':
        dq = quantized.DequantizeParams(
            quant = quantized.TORCH_DTYPE_TO_QUANT[x.dtype],
            shape = x.shape,
            dtype = x.dtype,
        )
        return TensorParams(data = x, dequant = dq)

    def tensors(self):
        yield self.data

    def get(self) -> Tensor:
        return self.dequant.apply(self.data)


@dataclass(frozen=True)
class EmbeddingParams:
    weight: TensorParams

    def tensors(self):
        yield from self.weight.tensors()

@dataclass(frozen=True)
class Embedding:
    def run(self, params: EmbeddingParams, x: Tensor) -> Tensor:
        return F.embedding(x, params.weight.get())


@dataclass(frozen=True)
class LinearParams:
    weight: TensorParams
    bias: Optional[TensorParams] = None

    def tensors(self):
        yield from self.weight.tensors()
        if self.bias is not None:
            yield from self.bias.tensors()

@dataclass(frozen=True)
class Linear:
    def run(self, params: LinearParams, x: Tensor) -> Tensor:
        bias = params.bias.get() if params.bias is not None else None
        return F.linear(x, params.weight.get(), bias)


@dataclass(frozen=True)
class RMSNormParams:
    scale: TensorParams

    def tensors(self):
        yield from self.scale.tensors()

@dataclass(frozen=True)
class RMSNorm:
    eps: float = 1e-6

    def run(self, params: RMSNormParams, x: Tensor) -> Tensor:
        scale = params.scale.get()
        assert scale.dtype == x.dtype, \
            'scale dtype %s should match input dtype %s' % (scale.dtype, x.dtype)
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * scale


@dataclass(frozen=True)
class CausalSelfAttentionParams:
    q_proj: Any
    k_proj: Any
    v_proj: Any
    output_proj: Any

    def tensors(self):
        yield from self.q_proj.tensors()
        yield from self.k_proj.tensors()
        yield from self.v_proj.tensors()
        yield from self.output_proj.tensors()

@dataclass(frozen=True)
class CausalSelfAttention:
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_proj: Any
    k_proj: Any
    v_proj: Any
    output_proj: Any
    pos_embeddings: nn.Module
    max_seq_len: int = 4096

    def run(
        self,
        params: CausalSelfAttentionParams,
        x: Tensor,
    ) -> Tensor:
        # input has shape [b, s, d]
        bsz, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # q has shape [b, s, num_heads * head_dim]
        # k has shape [b, s, num_kv_heads * head_dim]
        # v has shape [b, s, num_kv_heads * head_dim]
        q = self.q_proj.run(params.q_proj, x)
        k = self.k_proj.run(params.k_proj, x)
        v = self.v_proj.run(params.v_proj, x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads

        # q: [b, s, n_kv, q_per_kv, h_d]
        # k: [b, s, n_kv, 1, h_d]
        # v: [b, s, n_kv, 1, h_d]
        q = q.view(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)

        # if needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(q, input_pos=None)
        k = self.pos_embeddings(k, input_pos=None)

        # [b, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj.run(params.output_proj, output)


@dataclass(frozen=True)
class FeedForwardParams:
    gate_proj: Any
    down_proj: Any
    up_proj: Any

    def tensors(self):
        yield from self.gate_proj.tensors()
        yield from self.down_proj.tensors()
        yield from self.up_proj.tensors()

@dataclass(frozen=True)
class FeedForward:
    gate_proj: Any
    down_proj: Any
    up_proj: Any
    activation: nn.Module

    def run(self, params: FeedForwardParams, x: Tensor) -> Tensor:
        x1 = self.gate_proj.run(params.gate_proj, x)
        x3 = self.up_proj.run(params.up_proj, x)
        y = self.activation(x1) * x3
        return self.down_proj.run(params.down_proj, y)


@dataclass(frozen=True)
class TransformerDecoderLayerParams:
    attn: Any
    mlp: Any
    sa_norm: Any
    mlp_norm: Any

    def tensors(self):
        yield from self.attn.tensors()
        yield from self.mlp.tensors()
        yield from self.sa_norm.tensors()
        yield from self.mlp_norm.tensors()

@dataclass(frozen=True)
class TransformerDecoderLayer:
    attn: Any
    mlp: Any
    sa_norm: Any
    mlp_norm: Any

    def run(self, params: TransformerDecoderLayerParams, x: Tensor) -> Tensor:
        x_normed = self.sa_norm.run(params.sa_norm, x)
        attn_out = self.attn.run(params.attn, x_normed)
        h = attn_out + x
        h_normed = self.mlp_norm.run(params.mlp_norm, h)
        mlp_out = self.mlp.run(params.mlp, h_normed)
        out = h + mlp_out
        return out


@dataclass(frozen=True)
class TransformerDecoderParams:
    tok_embeddings: Any
    layers: List[Any]
    norm: Any
    output: Any

    def tensors(self):
        yield from self.tok_embeddings.tensors()
        for layer in self.layers:
            yield from layer.tensors()
        yield from self.norm.tensors()
        yield from self.output.tensors()

@dataclass(frozen=True)
class TransformerDecoder:
    tok_embeddings: Any
    layers: List[Any]
    norm: Any
    output: Any

    def run(self, params: TransformerDecoderParams, x: Tensor) -> Tensor:
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings.run(params.tok_embeddings, tokens)

        for layer, layer_params in zip(self.layers, params.layers, strict=True):
            # shape: [b, s, d]
            h = layer(layer_params, h)

        # shape: [b, s, d]
        h = self.norm(params.norm, h)

        # shape: [b, s, v]
        output = self.output(params.output, h).float()
        return output


@dataclass(frozen=True)
class WithAdapterParams:
    base: Any
    adapter: Any

    def tensors(self):
        yield from self.base.tensors()
        yield from self.adapter.tensors()

@dataclass(frozen=True)
class WithAdapter:
    base: Any
    adapter: Any

    def run(self, params: WithAdapterParams, x: Tensor) -> Tensor:
        y1 = self.base.run(params.base, x)
        y2 = self.adapter.run(params.adapter, x)
        return y1 + y2


@dataclass(frozen=True)
class LowRankAdapterParams:
    lora_a: TensorParams
    lora_b: TensorParams
    lora_alpha: float = 1.0

    def tensors(self):
        yield from self.lora_a.tensors()
        yield from self.lora_b.tensors()

@dataclass(frozen=True)
class LowRankAdapter:
    dropout: float = 0.0

    def run(self, params: LowRankAdapterParams, x: Tensor) -> Tensor:
        if self.dropout != 0.0:
            x = F.dropout(x, self.dropout)
        x = F.linear(x, params.lora_a.get())
        x = F.linear(x, params.lora_b.get())
        # https://arxiv.org/abs/2404.09610 "LoRA Dropout as a Sparsity
        # Regularizer for Overfitting Control" drops rows and columns in both A
        # and B, along the non-rank dimension only (to avoid reducing rank).
        # This is equivalent to applying dropout to both the input and output
        # vectors.
        if self.dropout != 0.0:
            x = F.dropout(x, self.dropout)
        return x * params.lora_alpha
    

@dataclass(frozen=True)
class EmbeddingLowRankAdapterParams:
    lora_a: TensorParams
    lora_b: TensorParams
    lora_alpha: float = 1.0

    def tensors(self):
        yield from self.lora_a.tensors()
        yield from self.lora_b.tensors()

@dataclass(frozen=True)
class EmbeddingLowRankAdapter:
    dropout: float = 0.0

    def run(self, params: EmbeddingLowRankAdapterParams, x: Tensor) -> Tensor:
        x = F.embedding(x, params.lora_a.get().t())
        x = F.linear(x, params.lora_b.get())
        if self.dropout != 0.0:
            x = F.dropout(x, self.dropout)
        return x * params.lora_alpha
