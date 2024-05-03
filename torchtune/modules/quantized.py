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

def _expand_bounds(bounds):
    """
    Given integer bounds, expand them to floating-point bounds such that all
    values within the floating-point bounds will round to an integer within the
    original bounds.
    """
    min_bound, max_bound = bounds
    return min_bound - 0.49, max_bound + 0.49

def latent_to_quantized(x, bounds):
    """
    Convert latent weights to quantized weights.  The output will consist
    entirely of integers within the `bounds`.
    """
    min_bound, max_bound = _expand_bounds(bounds)
    # `clamp` has zero derivative outside the bounds, which can cause weights
    # near the minimum/maximum to get "stuck".  We instead use `sigmoid` to do
    # the clamping.  It produces a value in the range `(0, 1)` and has nonzero
    # derivative everywhere.
    x = x.sigmoid()
    # Rescale to the expanded bounds.
    x = x * (max_bound - min_bound) + min_bound
    # Round, producing an integer within the original bounds.
    x = round_ste(x)
    return x

def quantized_to_latent(x, bounds):
    """
    Inverse of `latent_to_quantized`.  This function converts integer-valued
    quantized weights `x` to floating-point values `y` such that
    `latent_to_quantized(y, bounds)` produces `x`.
    """
    assert x.dtype.is_floating_point, 'must cast x to a floating point type first'

    min_bound, max_bound = _expand_bounds(bounds)

    # These are the inverses of the operations in `latent_to_quantized`,
    # applied in reverse order.

    # Ignore `round_ste`.  For a quantized weight of 3, there are many values
    # in the range `(2.5, 3.5)` that would round to the desired output.  We
    # just pick 3.0 for convenience.

    # Rescale from the expanded bounds to `(0, 1)`.
    x = (x - min_bound) / (max_bound - min_bound)

    # `logit` is the inverse of `sigmoid`.
    x = x.logit()

    return x


QS_BOUNDS = {
        # KSimple
        GGMLQuantizationType.Q3_K: (-4, 3),
        GGMLQuantizationType.Q6_K: (-32, 31),

        # KWithMin
        GGMLQuantizationType.Q2_K: (0, 3),
        GGMLQuantizationType.Q4_K: (0, 15),
        GGMLQuantizationType.Q5_K: (0, 31),
        }

SCALES_BOUNDS = {
        # KSimple
        GGMLQuantizationType.Q3_K: (-32, 31),
        GGMLQuantizationType.Q6_K: (-128, 127),

        # KWithMin
        GGMLQuantizationType.Q2_K: (0, 15),
        GGMLQuantizationType.Q4_K: (0, 63),
        GGMLQuantizationType.Q5_K: (0, 63),
        }

SUB_BLOCK_SHAPE = {
        # KSimple
        GGMLQuantizationType.Q3_K: (16, 16),
        GGMLQuantizationType.Q6_K: (16, 16),

        # KWithMin
        GGMLQuantizationType.Q2_K: (16, 16),
        GGMLQuantizationType.Q4_K: (8, 32),
        GGMLQuantizationType.Q5_K: (8, 32),
        }

# `dtype` to use for latent weights of various quant formats.
#
# TODO: Figure out whether bf16, fp16, or fp32 is best for the latent weights
# backing integer values.  I suspect we need high precision in the mantissa
# because we may need to accumulate many small updates into the latent weight
# before the integer value changes.  bf16 only has 8 bits of mantissa
# precision, so small updates applied to large values would be lost.
QUANTIZED_DTYPE = {
        # TODO: Could probably use bfloat16 for some of the smaller quants
        # TODO: Separate dtypes for `qs` and `scales`?
        GGMLQuantizationType.Q2_K: torch.float32,
        GGMLQuantizationType.Q3_K: torch.float32,
        GGMLQuantizationType.Q4_K: torch.float32,
        GGMLQuantizationType.Q5_K: torch.float32,
        GGMLQuantizationType.Q6_K: torch.float32,
        }

class QuantizedTensor_KSimple(nn.Module):
    """
    Zero-argument module that just reconstructs a tensor from its quantized
    form.  Supports "simple" K-quant formats: `qs * scales * d`.
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

        quant_dtype = QUANTIZED_DTYPE[self.quant]
        sub_blocks, elems = SUB_BLOCK_SHAPE[self.quant]

        self.k_qs = nn.Parameter(
                torch.empty((num_blocks, sub_blocks, elems), dtype=quant_dtype))
        self.k_scales = nn.Parameter(
                torch.empty((num_blocks, sub_blocks), dtype=quant_dtype))
        self.k_d = nn.Parameter(torch.empty((num_blocks,)))

        self.output_dtype = torch.get_default_dtype()

    @property
    def quant(self) -> GGMLQuantizationType:
        return GGMLQuantizationType(self._quant)

    def forward(self) -> Tensor:
        #assert not torch.isnan(self.k_qs).any(), 'got nan in INPUT qs'
        #assert not torch.isnan(self.k_scales).any(), 'got nan in INPUT scales'
        #assert not torch.isnan(self.k_d).any(), 'got nan in INPUT d'
        qs = latent_to_quantized(self.k_qs, QS_BOUNDS[self.quant])
        #assert not torch.isnan(qs).any(), 'got nan in qs'
        scales = latent_to_quantized(self.k_scales, SCALES_BOUNDS[self.quant])
        #assert not torch.isnan(scales).any(), 'got nan in scales'
        xs = (self.k_d.unsqueeze(1) * scales).unsqueeze(2) * qs
        #assert not torch.isnan(xs).any(), 'got nan in xs'
        return xs.view(-1)[:self.num_elems].view(self.shape).to(self.output_dtype)

class QuantizedTensor_KWithMin(nn.Module):
    """
    Zero-argument module that just reconstructs a tensor from its quantized
    form.  Supports K-quant formats with a minimum: `qs * sc * d - m * dmin`.
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

        quant_dtype = QUANTIZED_DTYPE[self.quant]
        sub_blocks, elems = SUB_BLOCK_SHAPE[self.quant]

        self.k_qs = nn.Parameter(
                torch.empty((num_blocks, sub_blocks, elems), dtype=quant_dtype))
        self.k_sc = nn.Parameter(
                torch.empty((num_blocks, sub_blocks), dtype=quant_dtype))
        self.k_m = nn.Parameter(
                torch.empty((num_blocks, sub_blocks), dtype=quant_dtype))
        self.k_d = nn.Parameter(torch.empty((num_blocks,)))
        self.k_dmin = nn.Parameter(torch.empty((num_blocks,)))

        self.output_dtype = torch.get_default_dtype()

    @property
    def quant(self) -> GGMLQuantizationType:
        return GGMLQuantizationType(self._quant)

    def forward(self) -> Tensor:
        #assert not torch.isnan(self.k_qs).any(), 'got nan in INPUT qs'
        #assert not torch.isnan(self.k_sc).any(), 'got nan in INPUT sc'
        #assert not torch.isnan(self.k_m).any(), 'got nan in INPUT m'
        #assert not torch.isnan(self.k_d).any(), 'got nan in INPUT d'
        #assert not torch.isnan(self.k_dmin).any(), 'got nan in INPUT dmin'
        qs = latent_to_quantized(self.k_qs, QS_BOUNDS[self.quant])
        #assert not torch.isnan(qs).any(), 'got nan in qs'
        sc = latent_to_quantized(self.k_sc, SCALES_BOUNDS[self.quant])
        #assert not torch.isnan(sc).any(), 'got nan in sc'
        scale = self.k_d.unsqueeze(1) * sc
        #assert not torch.isnan(scale).any(), 'got nan in scale'
        m = latent_to_quantized(self.k_m, SCALES_BOUNDS[self.quant])
        #assert not torch.isnan(m).any(), 'got nan in m'
        minimum = self.k_dmin.unsqueeze(1) * m
        #assert not torch.isnan(minimum).any(), 'got nan in minimum'
        xs = scale.unsqueeze(2) * qs - minimum.unsqueeze(2)
        #assert not torch.isnan(xs).any(), 'got nan in xs'
        ys = xs.view(-1)[:self.num_elems].view(self.shape).to(self.output_dtype)
        #assert not torch.isnan(ys).any(), 'got nan in ys'
        return xs.view(-1)[:self.num_elems].view(self.shape).to(self.output_dtype)

QUANTIZED_TENSOR_TYPE = {
        GGMLQuantizationType.Q3_K: QuantizedTensor_KSimple,
        GGMLQuantizationType.Q6_K: QuantizedTensor_KSimple,

        GGMLQuantizationType.Q2_K: QuantizedTensor_KWithMin,
        GGMLQuantizationType.Q4_K: QuantizedTensor_KWithMin,
        GGMLQuantizationType.Q5_K: QuantizedTensor_KWithMin,
        }

def make_quantized_tensor(shape, quant):
    type_ = QUANTIZED_TENSOR_TYPE[quant]
    return type_(shape, quant)

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
        self.weight_quant = make_quantized_tensor((out_features, in_features), weight_quant)
        if bias:
            self.bias_quant = make_quantized_tensor((out_features,), bias_quant)
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
