from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from torch import Tensor
from torch import nn
from typing import Optional, Any, Dict

from torchtune.modules import quantized
from torchtune.modules.attention import CausalSelfAttention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.rms_norm import RMSNorm
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings
from torchtune.modules.transformer import TransformerDecoderLayer


def make_linear(in_features, out_features, bias=True, *, weight_quant, bias_quant=None):
    need_quantized_weight = weight_quant not in quantized.UNQUANTIZED_TYPES
    need_quantized_bias = bias and bias_quant not in quantized.UNQUANTIZED_TYPES
    if need_quantized_weight or need_quantized_bias:
        return quantized.QuantLinear(in_features, out_features, bias=bias,
            weight_quant=weight_quant, bias_quant=bias_quant)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

def default_linear(name: str, in_features, out_features, bias=True):
    return nn.Linear(in_features, out_features, bias=bias)


# From torchtune.models.llama3._model_utils
def scale_hidden_dim_for_mlp(dim: int, multiple_of: int = 256) -> int:
    """Scale hidden dimension for MLP to keep number of parameters and computation constant.

    Args:
        dim (int): Input dimension.
        multiple_of (int): Round scaled dimension to nearest multiple of `multiple_of` for clean computation.

    Returns:
        Scaled hidden dimension.
    """
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * dim / 3)
    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

@dataclass(frozen=True)
class Llama3Arch:
    vocab_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    max_seq_len: int
    attn_dropout: float = 0.0
    rope_base: int = 500000.0
    intermediate_dim: Optional[int] = None
    norm_eps: float = 1e-5

    @staticmethod
    def llama3_8b():
        return Llama3Arch(
            vocab_size=128_256,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            embed_dim=4096,
            max_seq_len=8192,
            intermediate_dim=14336,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=500000.0,
        )

    @staticmethod
    def llama3_70b():
        return Llama3Arch(
            vocab_size=128_256,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            embed_dim=8192,
            max_seq_len=8192,
            intermediate_dim=28672,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=500000.0,
        )

    def make_module(self, key) -> nn.Module:
        kind, fmt = key
        quant_map = dict(fmt)
        def linear(name: str, in_features, out_features, bias=True):
            assert not bias
            rel_name = ('%s.weight' % name).split('.', 1)[1]
            weight_quant = quant_map[rel_name]
            need_quantized_weight = weight_quant not in quantized.UNQUANTIZED_TYPES
            if need_quantized_weight:
                return quantized.QuantLinear(in_features, out_features, bias=False,
                    weight_quant=weight_quant)
            else:
                return nn.Linear(in_features, out_features, bias=False)
        return self.make_module2(kind, linear=linear)

    def make_module2(self, kind: str, linear=default_linear) -> nn.Module:
        if kind == 'tok_embeddings':
            # TODO: Support quantized embedding
            return nn.Embedding(self.vocab_size, self.embed_dim)
        elif kind == 'layer':
            embed_dim = self.embed_dim
            if self.intermediate_dim is not None:
                hidden_dim = self.intermediate_dim
            else:
                hidden_dim = scale_hidden_dim_for_mlp(embed_dim)

            # From llama3._component_builders.llama3_mlp
            gate_proj = linear('layer.mlp.w1', embed_dim, hidden_dim, bias=False)
            down_proj = linear('layer.mlp.w2', hidden_dim, embed_dim, bias=False)
            up_proj = linear('layer.mlp.w3', embed_dim, hidden_dim, bias=False)
            mlp = FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)

            # From llama3._component_builders.llama3
            head_dim = embed_dim // self.num_heads
            rope = RotaryPositionalEmbeddings(
                dim=head_dim,
                max_seq_len=self.max_seq_len,
                base=self.rope_base,
            )
            self_attn = CausalSelfAttention(
                embed_dim=embed_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=head_dim,
                q_proj=linear('layer.attn.q_proj', embed_dim, self.num_heads * head_dim,
                    bias=False),
                k_proj=linear('layer.attn.k_proj', embed_dim, self.num_kv_heads * head_dim,
                    bias=False),
                v_proj=linear('layer.attn.v_proj', embed_dim, self.num_kv_heads * head_dim,
                    bias=False),
                output_proj=linear('layer.attn.output_proj', embed_dim, embed_dim,
                    bias=False),
                pos_embeddings=rope,
                max_seq_len=self.max_seq_len,
                attn_dropout=self.attn_dropout,
            )

            return TransformerDecoderLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=embed_dim, eps=self.norm_eps),
                mlp_norm=RMSNorm(dim=embed_dim, eps=self.norm_eps),
            )
        elif kind == 'norm':
            return RMSNorm(self.embed_dim, eps=self.norm_eps)
        elif kind == 'output':
            return linear('output', self.embed_dim, self.vocab_size, bias=False)
        else:
            assert False, 'bad module kind %r' % (kind,)
