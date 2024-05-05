"""
Measure error introduced at each layer by quantization.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
import math
import os
from pprint import pprint
import re
import sys
from tempfile import TemporaryDirectory
import time
from typing import Optional, Any, Dict, Tuple, List, Mapping
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from torchtune.models import convert_weights
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer_transformers
from torchtune.modules import quantized, lr_schedulers
from torchtune.modules.module_cache import ModuleCache, Llama3Arch
from torchtune.modules.tokenizers import Tokenizer
from torchtune.utils import FullModelGGUFCheckpointer, set_default_dtype
from gguf import GGUFReader, GGMLQuantizationType
from torchtune.utils import gguf_quant
import safetensors.torch
from tqdm import tqdm
from quant_repair.common import build_module, load_weights
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks
from quant_repair.datasets import load_slimorca_dataset
from quant_repair.weights import load_weights_safetensors_hf, \
    CheckpointStateDict, QuantizedCheckpointLoader


LAYER_PARAMETERS = {
    'attn.k_proj.weight',
    'attn.output_proj.weight',
    'attn.q_proj.weight',
    'attn.v_proj.weight',
    'mlp.w1.weight',
    'mlp.w2.weight',
    'mlp.w3.weight',
    'mlp_norm.scale',
    'sa_norm.scale',
}

CHECKPOINT_FILE_RE = re.compile(r'([a-z0-9_]+?)(_ckpt([0-9]+))?\.pt$')


def top_tokens(tokenizer, logits, top_k=3, temperature=1.0):
    # https://medium.com/@pashashaik/natural-language-generation-from-scratch-in-large-language-models-with-pytorch-4d9379635316
    #print('logits', logits.shape, logits)
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    top_k_probs = torch.nn.functional.softmax(top_k_logits / temperature, dim=-1)
    #print('top_k_logits', top_k_logits)
    #print('top_k_indices', top_k_indices)
    #print('top_k_probs', top_k_probs)
    return [(prob.item(), tokenizer.decode(token)) for prob, token in zip(top_k_probs, top_k_indices)]




@contextmanager
def record_time(dest, key):
    start = time.time()
    yield
    end = time.time()
    dest[key] += end - start


def main():
    with set_default_dtype(torch.bfloat16):
        with torch.no_grad():
            run()

def run():
    assert len(sys.argv) == 3
    orig_dir = sys.argv[1]
    quant_dir = sys.argv[2]

    device = torch.device('cuda')
    arch = Llama3Arch.llama3_8b()


    print('loading original weights from %r' % (orig_dir,))
    orig_state_dict = load_weights_safetensors_hf(orig_dir, arch)
    orig_weights = CheckpointStateDict(orig_state_dict)

    print('loading quantized weights from %r' % (quant_dir,))
    quant_weights = QuantizedCheckpointLoader(quant_dir, dequant_device=device)

    tokenizer_json_path = os.path.join(orig_dir, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    # Build forward-pass modules

    m_tok_embeddings = build_module(arch, 'tok_embeddings', device=device)
#    m_layer = arch.make_module2('layer', device='meta') \
#            .requires_grad_(False).to_empty(device=device)
#    m_layer.attn.pos_embeddings.reset_parameters()
#    m_layer.attn.pos_embeddings.to(device)
    m_layer = build_module(arch, 'layer', device=device)
    m_norm = build_module(arch, 'norm', device=device)
    m_output = build_module(arch, 'output', device=device)


    # Training config
    max_seq_len = 1024
    batch_size = 4
    total_epochs = 1
    max_steps_per_epoch = 160
    gradient_accumulation_steps = 2

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    superbatch_mem_gb = 32


    # Set up dataset and loader
    sampler, dataloader = load_slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
        seed=12345,
        batch_size=batch_size,
    )


    # Loss function
    loss_fn = nn.MSELoss()
    loss_sum = [0.] * arch.num_layers
    loss_count = [0] * arch.num_layers

    kldiv_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')
    kldiv_sum = 0.
    kldiv_count = 0


    # Training loop
    for curr_epoch in range(0, total_epochs):
        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        sampler.set_epoch(curr_epoch)

        samples_iter = (tokens for tokens, labels in dataloader)
        samples_iter = itertools.islice(samples_iter, max_samples)
        samples_iter = iter(pbar := tqdm(samples_iter,
            total=max_samples, desc='samples', position=0, leave=True))
        samples_processed = 0

        embeds_orig = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb // 2)
        embeds_quant = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb // 2)
        superbatch_limit = embeds_orig.tokens_free()

        for superbatch_samples in sized_chunks(samples_iter, superbatch_limit,
                lambda t: t.numel()):
            embeds_orig.clear()
            embeds_quant.clear()

            layers_iter = tqdm(range(arch.num_layers), desc='layers', position=1, leave=False)
            superbatch_tqdm_kwargs = dict(desc='superbatch', position=2, leave=False)

            # Process tok_embeddings modules
            load_weights(orig_weights, m_tok_embeddings, 'tok_embeddings')
            for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                y = m_tok_embeddings(sample.to(device))
                embeds_orig.append(y)

            load_weights(quant_weights, m_tok_embeddings, 'tok_embeddings')
            for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                y = m_tok_embeddings(sample.to(device))
                embeds_quant.append(y)

            # Process layers
            for layer_index in layers_iter:
                load_weights(orig_weights, m_layer, 'layers.%d' % layer_index)
                embeds_orig.apply(m_layer, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)
                #tqdm.write('orig %d: %s' % (layer_index, embeds_orig[0][0]))

                load_weights(quant_weights, m_layer, 'layers.%d' % layer_index)
                embeds_quant.apply(m_layer, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)
                #tqdm.write('quant %d: %s' % (layer_index, embeds_quant[0][0]))

                for orig_output, quant_output in zip(embeds_orig, embeds_quant):
                    loss = loss_fn(quant_output.to(device), orig_output.to(device))
                    #loss = loss / gradient_accumulation_steps
                    loss_sum[layer_index] += loss.item()
                    loss_count[layer_index] += 1

                tqdm.write('layer %d: loss = %.6e' %
                    (layer_index, loss_sum[layer_index] / loss_count[layer_index]))
                tqdm.write('  sum %.6e, count %d' %
                    (loss_sum[layer_index], loss_count[layer_index]))

            for orig_batch, quant_batch in tqdm(zip(embeds_orig, embeds_quant),
                    total=len(embeds_orig), **superbatch_tqdm_kwargs):
                load_weights(orig_weights, m_norm, 'norm')
                load_weights(orig_weights, m_output, 'output')
                orig_logits = m_output(m_norm(orig_batch.to(device)))

                load_weights(quant_weights, m_norm, 'norm')
                load_weights(quant_weights, m_output, 'output')
                quant_logits = m_output(m_norm(quant_batch.to(device)))

                orig_log_prob = torch.nn.functional.log_softmax(orig_logits, dim=-1)
                quant_log_prob = torch.nn.functional.log_softmax(quant_logits, dim=-1)

                #tqdm.write(str(orig_logits.shape))
                #for ii in range(8):
                #    tqdm.write(str(top_tokens(tokenizer, orig_logits[ii][200], top_k=5)))
                #    tqdm.write(str(top_tokens(tokenizer, quant_logits[ii][200], top_k=5)))
                #    tqdm.write('token kldiv: %s' % torch.nn.functional.kl_div(
                #        quant_logits[ii][200], orig_logits[ii][200],
                #        reduction='batchmean', log_target=True))
                #    tqdm.write('token kldiv alt: %s' % torch.nn.functional.kl_div(
                #        quant_log_prob[ii][200], orig_log_prob[ii][200],
                #        reduction='batchmean', log_target=True))
                #    tqdm.write('token kldiv alt2: %s' % torch.nn.functional.kl_div(
                #        quant_log_prob[ii][200], orig_log_prob[ii][200],
                #        reduction='mean', log_target=True))
                #    tqdm.write('prompt kldiv: %s' % torch.nn.functional.kl_div(
                #        quant_logits[ii], orig_logits[ii],
                #        reduction='batchmean', log_target=True))
                #    tqdm.write('prompt kldiv alt: %s' % torch.nn.functional.kl_div(
                #        quant_log_prob[ii], orig_log_prob[ii],
                #        reduction='batchmean', log_target=True))
                #    tqdm.write('prompt kldiv alt2: %s' % torch.nn.functional.kl_div(
                #        quant_log_prob[ii], orig_log_prob[ii],
                #        reduction='mean', log_target=True))
                #tqdm.write('batch kldiv: %s' % torch.nn.functional.kl_div(
                #    quant_logits, orig_logits,
                #    reduction='batchmean', log_target=True))
                #tqdm.write('batch kldiv alt: %s' % torch.nn.functional.kl_div(
                #    quant_log_prob, orig_log_prob,
                #    reduction='batchmean', log_target=True))
                #tqdm.write('batch kldiv alt2: %s' % torch.nn.functional.kl_div(
                #    quant_log_prob, orig_log_prob,
                #    reduction='mean', log_target=True))
                #tqdm.write('shape = %s' % (quant_log_prob.shape,))
                #tqdm.write('batch kldiv alt3: %s' % torch.nn.functional.kl_div(
                #    quant_log_prob.view(-1, 128_256), orig_log_prob.view(-1, 128_256),
                #    reduction='batchmean', log_target=True))

                orig_log_prob = orig_log_prob.view(-1, arch.vocab_size)
                quant_log_prob = quant_log_prob.view(-1, arch.vocab_size)
                num_tokens = orig_log_prob.shape[0]

                kldiv = kldiv_fn(quant_log_prob, orig_log_prob)
                #tqdm.write('quant: %s' % quant_logits[0])
                #tqdm.write('orig: %s' % orig_logits[0])
                tqdm.write('kldiv: %s' % kldiv)
                kldiv_sum += kldiv.item() * num_tokens
                kldiv_count += num_tokens
            tqdm.write('kldiv: loss = %.6e' % (kldiv_sum / kldiv_count))
            tqdm.write('  sum %.6e, count %d' % (kldiv_sum, kldiv_count))


    print('total loss:')
    for layer_index in layers_iter:
        print('layer %d: loss = %.6e' %
            (layer_index, loss_sum[layer_index] / loss_count[layer_index]))
        print('  sum %.6e, count %d' % (loss_sum[layer_index], loss_count[layer_index]))
    print('kldiv: loss = %.6e' % (kldiv_sum / kldiv_count))
    print('  sum %.6e, count %d' % (kldiv_sum, kldiv_count))


if __name__ == '__main__':
    main()
