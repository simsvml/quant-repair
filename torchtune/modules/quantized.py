import torch

from torch import nn, Tensor

from gguf import GGMLQuantizationType


QK_K = 256

def round_ste(t: Tensor) -> Tensor:
    """
    Like `t.round()`, but with a straight-through estimator for the backward
    pass.
    """
    return t + (t.round() - t).detach()

QS_BOUNDS = {
        GGMLQuantizationType.Q6_K: (-32, 31),
        }

SCALES_BOUNDS = {
        GGMLQuantizationType.Q6_K: (-128, 127),
        }

class QuantizedTensor(nn.Module):
    """
    Zero-argument module that just reconstructs a tensor from its quantized
    form.
    """

    def __init__(self, shape, quant) -> None:
        super().__init__()
        self.shape = shape
        self._quant = int(quant)

        num_elems = 1
        for d in shape:
            num_elems *= d
        self.num_elems = num_elems
        num_blocks = num_elems // QK_K

        # TODO: Figure out whether bf16, fp16, or fp32 is best for the latent
        # weights backing integer values.  I suspect we need high precision in
        # the mantissa because we may need to accumulate many small updates
        # into the latent weight before the integer value changes.  bf16 only
        # has 8 bits of mantissa precision, so small updates applied to large
        # values would be lost.
        quant_dtype = torch.float32
        #quant_dtype = torch.float16
        #quant_dtype = torch.bfloat16

        # TODO: 16x16 is used for Q6_K; might need to vary this for other modes
        self.k_qs = nn.Parameter(
                torch.empty((num_blocks, 16, 16), dtype=quant_dtype))
        self.k_scales = nn.Parameter(
                torch.empty((num_blocks, 16), dtype=quant_dtype))
        self.k_d = nn.Parameter(torch.empty((num_blocks,)))

        self.output_dtype = torch.get_default_dtype()

    @property
    def quant(self) -> GGMLQuantizationType:
        return GGMLQuantizationType(self._quant)

    def forward(self) -> Tensor:
        qs = round_ste(self.k_qs).clamp(*QS_BOUNDS[self.quant])
        #assert not torch.isnan(qs).any(), 'got nan in qs'
        scales = round_ste(self.k_scales).clamp(*SCALES_BOUNDS[self.quant])
        #assert not torch.isnan(scales).any(), 'got nan in scales'
        #assert not torch.isnan(self.k_d).any(), 'got nan in d'
        xs = (self.k_d.unsqueeze(1) * scales).unsqueeze(2) * qs
        #assert not torch.isnan(xs).any(), 'got nan in xs'
        return xs.view(-1)[:self.num_elems].view(self.shape).to(self.output_dtype)

class QuantLinear(nn.Module):
    """
    Quantized version of `nn.Linear`.
    """

    def __init__(
        self, in_features, out_features, bias=True, *, weight_quant, bias_quant=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant = QuantizedTensor((out_features, in_features), weight_quant)
        if bias:
            self.bias_quant = QuantizedTensor((out_features,), bias_quant)
        else:
            self.register_parameter('bias_quant', None)

    @classmethod
    def from_module(self, m, weight_quant, bias_quant=None):
        return QuantLinear(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight_quant.forward()
        #assert not torch.isnan(weight).any(), 'got nan in weight'
        bias = self.bias_quant.forward() if self.bias_quant is not None else None
        #assert bias is None or not torch.isnan(bias).any(), 'got nan in bias'
        return nn.functional.linear(x, weight, bias)


UNQUANTIZED_TYPES = {
        GGMLQuantizationType.F16,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F64,
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
        }

def replace_modules(module, quant_map):
    """
    Replace descendants of `module` with quantized equivalents where possible.
    `quant_map` is used to determine the quantization type to use in each
    replacement.
    """
    replacement_map = {}
    def walk(parent, prefix):
        for child_name, child in parent.named_children():
            if id(child) in replacement_map:
                # We don't have to worry about a newly-allocated module
                # colliding with an old `id(child)` because we only inspect old
                # modules during this traversal.
                parent.add_module(child_name, replacement_map[id(child)])
            elif isinstance(child, nn.Linear):
                weight_quant = quant_map['%s%s.weight' % (prefix, child_name)]
                bias_quant = quant_map.get('%s%s.bias' % (prefix, child_name))
                if weight_quant in UNQUANTIZED_TYPES or bias_quant in UNQUANTIZED_TYPES:
                    continue
                print('replace %s%s using %s, %s' % (prefix, child_name, weight_quant, bias_quant))
                new_child = QuantLinear.from_module(child, weight_quant, bias_quant)
                replacement_map[id(child)] = new_child
                parent.add_module(child_name, new_child)
            else:
                walk(child, '%s%s.' % (prefix, child_name))
    walk(module, '')
