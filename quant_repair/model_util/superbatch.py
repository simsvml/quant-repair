from typing import Any, List, Callable
import torch
from torch import Tensor
from .. import functional as QRF
from .. import model_util as QRM
from ..forward import SuperbatchEmbeddings

@torch.no_grad()
def run_forward_superbatch(
    model: QRF.TransformerDecoder,
    loader,
    samples: List[Tensor],
    embeds: SuperbatchEmbeddings,
    device,
    pbar_forward = None,
    pbar_layer = None,
):
    """
    Process `samples` through all but the output layer of `model`, storing the
    results in `embeds`.  Uses `loader` to load the weights for each layer, one
    at a time.
    """
    run_forward_superbatch2(
        samples,
        embeds,
        len(model.layers),
        lambda: QRM.llama3.build_forward_tok_embeddings(model, loader, device),
        lambda i: QRM.llama3.build_forward_layer(model, loader, i, device),
        lambda: QRM.llama3.build_forward_norm(model, loader, device),
        device,
        pbar_forward = pbar_forward,
        pbar_layer = pbar_layer,
    )

@torch.no_grad()
def run_forward_superbatch2(
    samples: List[Tensor],
    embeds: SuperbatchEmbeddings,
    num_layers: int,
    build_tok_embeddings: Callable[[], Any],
    build_layer: Callable[[int], Any],
    build_norm: Callable[[], Any],
    device,
    pbar_forward = None,
    pbar_layer = None,
):
    """
    Process `samples` through all but the output layer of `model`, storing the
    results in `embeds`.  Uses `loader` to load the weights for each layer, one
    at a time.
    """
    embeds.clear()

    if len(samples) == 0:
        return

    if pbar_forward is not None:
        pbar_forward.reset(2 + num_layers)

    # tok_embeddings
    m = build_tok_embeddings()
    if pbar_layer is not None:
        pbar_layer.reset(len(samples))
    for sample in samples:
        y = m(sample.to(device))
        embeds.append(y)
        if pbar_layer is not None:
            pbar_layer.update()
    del m, sample, y
    if pbar_forward is not None:
        pbar_forward.update()

    # layers
    for i in range(num_layers):
        m = build_layer(i)
        embeds.apply(m, device = device, pbar = pbar_layer)
        del m
        if pbar_forward is not None:
            pbar_forward.update()

    # norm
    m = build_norm()
    embeds.apply(m, device = device, pbar = pbar_layer)
    del m
    if pbar_forward is not None:
        pbar_forward.update()
