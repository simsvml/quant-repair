from dataclasses import dataclass
from typing import List, Callable
from torch import Tensor
from .. import functional as QRF
from ..memory_accounting import MEMORY_ACCOUNTING


@dataclass(frozen=True)
class LayerTrainableParams:
    q_proj: QRF.LowRankAdapterParams
    k_proj: QRF.LowRankAdapterParams
    v_proj: QRF.LowRankAdapterParams
    output_proj: QRF.LowRankAdapterParams
    gate_proj: QRF.LowRankAdapterParams
    down_proj: QRF.LowRankAdapterParams
    up_proj: QRF.LowRankAdapterParams
    sa_norm: QRF.RMSNormParams
    mlp_norm: QRF.RMSNormParams

    def tensors(self):
        yield from self.q_proj.tensors()
        yield from self.k_proj.tensors()
        yield from self.v_proj.tensors()
        yield from self.output_proj.tensors()
        yield from self.gate_proj.tensors()
        yield from self.down_proj.tensors()
        yield from self.up_proj.tensors()
        yield from self.sa_norm.tensors()
        yield from self.mlp_norm.tensors()

@dataclass(frozen=True)
class TrainableParams:
    tok_embeddings: QRF.LowRankAdapterParams
    layers: List[LayerTrainableParams]
    norm: QRF.RMSNormParams
    output: QRF.LowRankAdapterParams

    def tensors(self):
        yield from self.tok_embeddings.tensors()
        for layer in self.layers:
            yield from layer.tensors()
        yield from self.norm.tensors()
        yield from self.output.tensors()


def weights_getter(loader, device):
    def get1(key):
        return loader.get(key, dequant=True)[key].to(device)
    return get1


def build_trainable_tok_embeddings(
    model: QRF.TransformerDecoder,
    loader,
    train_params: TrainableParams,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    params = QRF.WithAdapterParams(
        base = QRF.LinearParams(get1('tok_embeddings.weight')),
        adapter = train_params.tok_embeddings,
    )
    MEMORY_ACCOUNTING.register_params(params, 'build_trainable_tok_embeddings params')
    def run(x):
        return model.tok_embeddings.run(params, x)
    return run

def build_trainable_layer(
    model: QRF.TransformerDecoder,
    loader,
    train_params: TrainableParams,
    layer_index: int,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    train_layer_params = train_params.layers[layer_index]
    params = QRF.TransformerDecoderLayerParams(
        attn = QRF.CausalSelfAttentionParams(
            q_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.attn.q_proj.weight' % layer_index)),
                adapter = train_layer_params.q_proj,
            ),
            k_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.attn.k_proj.weight' % layer_index)),
                adapter = train_layer_params.k_proj,
            ),
            v_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.attn.v_proj.weight' % layer_index)),
                adapter = train_layer_params.v_proj,
            ),
            output_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.attn.output_proj.weight' % layer_index)),
                adapter = train_layer_params.output_proj,
            ),
        ),
        mlp = QRF.FeedForwardParams(
            gate_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.mlp.w1.weight' % layer_index)),
                adapter = train_layer_params.gate_proj,
            ),
            down_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.mlp.w2.weight' % layer_index)),
                adapter = train_layer_params.down_proj,
            ),
            up_proj = QRF.WithAdapterParams(
                base = QRF.LinearParams(get1('layers.%d.mlp.w3.weight' % layer_index)),
                adapter = train_layer_params.up_proj,
            ),
        ),
        sa_norm = train_layer_params.sa_norm,
        mlp_norm = train_layer_params.mlp_norm,
    )
    MEMORY_ACCOUNTING.register_params(params, 'build_trainable_layer params')
    def run(x):
        return model.layers[layer_index].run(params, x)
    return run

def build_trainable_norm(
    model: QRF.TransformerDecoder,
    loader,
    train_params: TrainableParams,
    device,
) -> Callable[[Tensor], Tensor]:
    params = train_params.norm
    MEMORY_ACCOUNTING.register_params(params, 'build_trainable_norm params')
    def run(x):
        return model.norm.run(params, x)
    return run

def build_trainable_output(
    model: QRF.TransformerDecoder,
    loader,
    train_params: TrainableParams,
    device,
) -> Callable[[Tensor], Tensor]:
    get1 = weights_getter(loader, device)
    params = QRF.WithAdapterParams(
        base = QRF.LinearParams(get1('output.weight')),
        adapter = train_params.output,
    )
    MEMORY_ACCOUNTING.register_params(params, 'build_trainable_output params')
    def run(x):
        return model.output.run(params, x)
    return run
