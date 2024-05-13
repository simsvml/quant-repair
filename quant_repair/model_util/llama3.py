from typing import Callable
from torch import Tensor
from .. import functional as QRF
from ..memory_accounting import MEMORY_ACCOUNTING
from .misc import weights_getter


def build_forward_tok_embeddings(
    model: QRF.TransformerDecoder,
    loader,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    params = QRF.LinearParams(get1('tok_embeddings.weight'))
    MEMORY_ACCOUNTING.register_params(params, 'build_forward_tok_embeddings params')
    def run(x):
        return model.tok_embeddings.run(params, x)
    return run

def build_forward_layer(
    model: QRF.TransformerDecoder,
    loader,
    layer_index: int,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    i = layer_index
    params = QRF.TransformerDecoderLayerParams(
        attn = QRF.CausalSelfAttentionParams(
            q_proj = QRF.LinearParams(get1('layers.%d.attn.q_proj.weight' % i)),
            k_proj = QRF.LinearParams(get1('layers.%d.attn.k_proj.weight' % i)),
            v_proj = QRF.LinearParams(get1('layers.%d.attn.v_proj.weight' % i)),
            output_proj = QRF.LinearParams(get1('layers.%d.attn.output_proj.weight' % i)),
        ),
        mlp = QRF.FeedForwardParams(
            gate_proj = QRF.LinearParams(get1('layers.%d.mlp.w1.weight' % i)),
            down_proj = QRF.LinearParams(get1('layers.%d.mlp.w2.weight' % i)),
            up_proj = QRF.LinearParams(get1('layers.%d.mlp.w3.weight' % i)),
        ),
        sa_norm = QRF.RMSNormParams(get1('layers.%d.sa_norm.scale' % i)),
        mlp_norm = QRF.RMSNormParams(get1('layers.%d.mlp_norm.scale' % i)),
    )
    MEMORY_ACCOUNTING.register_params(params, 'build_forward_layer params')
    def run(x):
        return model.layers[layer_index].run(params, x)
    return run

def build_forward_norm(
    model: QRF.TransformerDecoder,
    loader,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    params = QRF.RMSNormParams(get1('norm.scale'))
    MEMORY_ACCOUNTING.register_params(params, 'build_forward_norm params')
    def run(x):
        return model.norm.run(params, x)
    return run

def build_forward_output(
    model: QRF.TransformerDecoder,
    loader,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    params = QRF.LinearParams(get1('output.weight'))
    MEMORY_ACCOUNTING.register_params(params, 'build_forward_output params')
    def run(x):
        return model.output.run(params, x)
    return run
