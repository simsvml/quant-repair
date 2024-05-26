from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from gguf import GGMLQuantizationType
import torch_ggml_quant


UNQUANTIZED_TYPES = {
        GGMLQuantizationType.F16,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F64,
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
        }


@dataclass(frozen=True)
class DequantizeParams:
    quant: GGMLQuantizationType
    shape: Tuple
    dtype: Optional[torch.dtype] = None

    def apply(self, x: Tensor) -> Tensor:
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()

        if self.quant in UNQUANTIZED_TYPES:
            assert x.shape == self.shape
            return x.to(dtype)

        x = x.view(-1, torch_ggml_quant.quant_format_block_size(self.quant))
        k = torch_ggml_quant.quant_format_values_per_block(self.quant)
        buf_shape = (x.shape[0], k)

        if x.device.type == 'cpu':
            buf = torch.empty(buf_shape, device=x.device, dtype=torch.float32)
            torch_ggml_quant.dequantize_fp32_cpu(x, buf, self.quant)
        elif x.device.type == 'cuda':
            if self.dtype == torch.float32:
                buf = torch.empty(buf_shape, device=x.device, dtype=torch.float32)
                torch_ggml_quant.dequantize_fp32_cuda(x, buf, self.quant)
            else:
                buf = torch.empty(buf_shape, device=x.device, dtype=torch.float16)
                torch_ggml_quant.dequantize_fp16_cuda(x, buf, self.quant)
        else:
            raise AssertionError(
                'GGML dequantization is only supported on cpu and cuda, not %s' % x.device)

        row_dim = self.shape[-1]
        padded_row_dim = (row_dim + k - 1) // k * k
        y = buf.view(self.shape[:-1] + (padded_row_dim,))
        y = y[..., 0:row_dim]
        return y.to(dtype)

# Mostly copied from the `LinearFunction` examples in the PyTorch docs:
# http://pytorch.org/docs/master/notes/extending.html
class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        x,
        weight,
        weight_dequant,
        bias,
        bias_dequant,
    ):
        #print('x shape', x.shape, 'weight shape', weight_dequant.shape)
        x_shape = x.shape
        x = x.view(-1, weight_dequant.shape[1])
        y = x.mm(weight_dequant.apply(weight).t())
        if bias is not None:
            y += bias_dequant.apply(bias).unsqueeze(0).expand_as(output)
        #print('linear: input = %s, weight = %s, output = %s' %
        #      (x.dtype, weight_dequant.dtype or torch.get_default_dtype(), y.dtype))
        y = y.view(*x_shape[:-1], weight_dequant.shape[0])
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, weight_dequant, bias, bias_dequant = inputs
        ctx.save_for_backward(x, weight)
        ctx.weight_dequant = weight_dequant
        ctx.has_bias = bias is not None

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        weight_dequant = ctx.weight_dequant
        has_bias = ctx.has_bias
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output \
                .view(-1, weight_dequant.shape[0]) \
                .mm(weight_dequant.apply(weight)) \
                .view(*grad_output.shape[:-1], weight_dequant.shape[1])
        if ctx.needs_input_grad[1]:
            #grad_weight = grad_output.t().mm(x)
            assert False, 'TODO: Implement QuantLinearFunction backward pass for weight'
        if has_bias and ctx.needs_input_grad[2]:
            #grad_bias = grad_output.sum(0)
            assert False, 'TODO: Implement QuantLinearFunction backward pass for bias'

        return grad_input, grad_weight, None, grad_bias, None

def quant_linear(
    x: Tensor,
    weight: Tensor,
    weight_dequant: DequantizeParams,
    bias: Optional[Tensor] = None,
    bias_dequant: Optional[DequantizeParams] = None,
) -> Tensor:
    return QuantLinearFunction.apply(x, weight, weight_dequant, bias, bias_dequant)


class QuantEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        x,
        weight,
        weight_dequant,
    ):
        # Select quantized rows of `weight` at indices `x`
        vocab_size, embed_dim = weight_dequant.shape
        block_size = torch_ggml_quant.quant_format_block_size(weight_dequant.quant)
        k = torch_ggml_quant.quant_format_values_per_block(weight_dequant.quant)

        row_blocks = (embed_dim + k - 1) // k
        row_bytes = row_blocks * block_size
        weight = weight.view(vocab_size, row_bytes)
        y_quant = F.embedding(x, weight)

        # Dequantize the rows
        y_dequant = DequantizeParams(
            quant = weight_dequant.quant,
            shape = y_quant.shape[:-1] + (embed_dim,),
            dtype = weight_dequant.dtype,
        )
        y = y_dequant.apply(y_quant)
        #print('embedding: input = %s, weight = %s, output = %s' %
        #      (x.dtype, weight_dequant.dtype or torch.get_default_dtype(), y.dtype))
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, weight_dequant = inputs
        ctx.save_for_backward(x, weight)
        ctx.weight_dequant = weight_dequant

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        weight_dequant = ctx.weight_dequant
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            #grad_input = grad_output.mm(weight_dequant.apply(weight))
            assert False, 'TODO: Implement QuantEmbeddingFunction backward pass'
        if ctx.needs_input_grad[1]:
            #grad_weight = grad_output.t().mm(x)
            assert False, 'TODO: Implement QuantEmbeddingFunction backward pass'

        return grad_input, grad_weight

def quant_embedding(
    x: Tensor,
    weight: Tensor,
    weight_dequant: DequantizeParams,
) -> Tensor:
    return QuantEmbeddingFunction.apply(x, weight, weight_dequant)
