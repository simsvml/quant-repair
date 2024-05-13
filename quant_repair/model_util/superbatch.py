import torch
from .. import functional as QRF
from .. import model_util as QRM
from ..forward import SuperbatchEmbeddings
from .misc import weights_getter

@torch.no_grad()
def run_forward_superbatch(
    model: QRF.TransformerDecoder,
    loader,
    samples,
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
    get1 = weights_getter(loader, device)

    embeds.clear()
    if pbar_forward is not None:
        pbar_forward.reset(2 + len(model.layers))

    # tok_embeddings
    m = QRM.llama3.build_forward_tok_embeddings(model, loader, device)
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
    for i in range(len(model.layers)):
        m = QRM.llama3.build_forward_layer(model, loader, i, device)
        embeds.apply(m, device = device, pbar = pbar_layer)
        del m
        if pbar_forward is not None:
            pbar_forward.update()

    # norm
    m = QRM.llama3.build_forward_norm(model, loader, device)
    embeds.apply(m, device = device, pbar = pbar_layer)
    del m
    if pbar_forward is not None:
        pbar_forward.update()
