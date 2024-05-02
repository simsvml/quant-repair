"""
Measure error introduced at each layer by quantization.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
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
from torchtune.datasets import slimorca_dataset
from torchtune.utils import FullModelGGUFCheckpointer, set_default_dtype
from gguf import GGUFReader, GGMLQuantizationType
from torchtune.utils import gguf_quant
from torchtune.utils import padded_collate
import safetensors.torch
from tqdm import tqdm
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks


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




class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        tokens = self._tokenizer.encode(sample['text'], add_bos=True, add_eos=False)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        #labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        #assert len(tokens) == len(labels)
        labels = tokens

        return tokens, labels


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
    orig_state_dict = {}
    for name in os.listdir(orig_dir):
        if not (name.startswith('model') and name.endswith('.safetensors')):
            continue
        # This will mmap the file.  Data will be paged in and out on
        # demand, so this effectively consumes no RAM.
        path = os.path.join(orig_dir, name)
        chunk = safetensors.torch.load_file(path, device='cpu')
        for key in chunk.keys():
            assert key not in orig_state_dict, \
                'duplicate tensor %s found in %s' % (key, path)
        orig_state_dict.update(chunk)
    orig_state_dict = convert_weights.hf_to_tune(
        orig_state_dict,
        num_heads=arch.num_heads,
        num_kv_heads=arch.num_kv_heads,
        dim=arch.embed_dim,
    )


    print('loading quantized weights from %r' % (quant_dir,))
    # Find the newest checkpoint for each module.  We don't load these files
    # until the data is needed.
    quant_module_files = {}
    for name in os.listdir(quant_dir):
        match = CHECKPOINT_FILE_RE.match(name)
        if match is None:
            continue
        key = match.group(1)
        version = match.group(3)
        if version is None:
            version = -1
        else:
            version = int(version)
        if key not in quant_module_files:
            insert = True
        else:
            old_version = quant_module_files[key][1]
            insert = version > old_version
        if insert:
            quant_module_files[key] = (name, version)

    quant_map = torch.load(os.path.join(quant_dir, 'quant_map.pt'))
    quant_shape_map = torch.load(os.path.join(quant_dir, 'quant_shape_map.pt'))

    _LAST_QUANT_FILE_PATH = None
    _LAST_QUANT_STATE_DICT = None
    def get_dequantized_weights(name):
        nonlocal _LAST_QUANT_FILE_PATH, _LAST_QUANT_STATE_DICT
        quant, shape = quant_shape_map[name]

        if name.startswith('layers.'):
            prefix, layer_index_str, rel_name = name.split('.', 2)
            layer_index = int(layer_index_str)
            module_key = 'layer%d' % layer_index
        else:
            module_key, rel_name = name.split('.', 1)

        file_name, _ = quant_module_files[module_key]
        file_path = os.path.join(quant_dir, file_name)
        if file_path == _LAST_QUANT_FILE_PATH:
            quant_state_dict = _LAST_QUANT_STATE_DICT
        else:
            tqdm.write('load %s' % file_path)
            quant_state_dict = torch.load(file_path)
            _LAST_QUANT_STATE_DICT = quant_state_dict
            _LAST_QUANT_FILE_PATH = file_path

        tqdm.write('get_dequantized_weights(%s)' % name)

        if quant in quantized.UNQUANTIZED_TYPES or module_key == 'tok_embeddings':
            return quant_state_dict[rel_name]
        else:
            m = quantized.make_quantized_tensor(shape, quant)
            m.load_state_dict({
                param_name: quant_state_dict['%s_quant.%s' % (rel_name, param_name)]
                for param_name, _ in m.named_parameters()
            })
            return m.forward()


    tokenizer_json_path = os.path.join(orig_dir, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    # Build forward-pass modules
    m_tok_embeddings = arch.make_module(('tok_embeddings',
        {'weight': GGMLQuantizationType.F16})).to(device)
    m_layer = arch.make_module(('layer',
        {name: GGMLQuantizationType.F16 for name in LAYER_PARAMETERS})).to(device)

    m_norm_orig = arch.make_module(('norm',
        {'scale': GGMLQuantizationType.F16})).to(device)
    m_output_orig = arch.make_module(('output',
        {'weight': GGMLQuantizationType.F16})).to(device)
    m_norm_quant = arch.make_module(('norm',
        {'scale': GGMLQuantizationType.F16})).to(device)
    m_output_quant = arch.make_module(('output',
        {'weight': GGMLQuantizationType.F16})).to(device)

    m_norm_orig.load_state_dict({
        'scale': orig_state_dict['norm.scale'],
    })
    m_output_orig.load_state_dict({
        'weight': orig_state_dict['output.weight'],
    })

    m_norm_quant.load_state_dict({
        'scale': get_dequantized_weights('norm.scale'),
    })
    m_output_quant.load_state_dict({
        'weight': get_dequantized_weights('output.weight'),
    })


    # Training config
    max_seq_len = 1024
    batch_size = 8
    total_epochs = 1
    max_steps_per_epoch = 1
    gradient_accumulation_steps = 1

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    superbatch_mem_gb = 32


    # Set up dataset and loader
    dataset = slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
    )
#    dataset = TextDataset(
#        tokenizer=tokenizer,
#        max_seq_len=max_seq_len,
#        source='wikitext',
#        name='wikitext-2-raw-v1',
#        split='test',
#    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
    )
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=partial(
            padded_collate,
            padding_idx=tokenizer.pad_id,
            #ignore_idx=loss_fn.ignore_index,
            ignore_idx=-100,
        ),
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
            m_tok_embeddings.load_state_dict({
                'weight': orig_state_dict['tok_embeddings.weight'],
            })
            for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                y = m_tok_embeddings(sample.to(device))
                embeds_orig.append(y)

            m_tok_embeddings.load_state_dict({
                'weight': get_dequantized_weights('tok_embeddings.weight'),
            })
            for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                y = m_tok_embeddings(sample.to(device))
                embeds_quant.append(y)

            # Process layers
            for layer_index in layers_iter:
                m_layer.load_state_dict({
                    name: orig_state_dict['layers.%d.%s' % (layer_index, name)]
                    for name in LAYER_PARAMETERS
                })
                embeds_orig.apply(m_layer, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)
                tqdm.write('orig %d: %s' % (layer_index, embeds_orig[0][0]))

                m_layer.load_state_dict({
                    name: get_dequantized_weights('layers.%d.%s' % (layer_index, name))
                    for name in LAYER_PARAMETERS
                })
                embeds_quant.apply(m_layer, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)
                tqdm.write('quant %d: %s' % (layer_index, embeds_quant[0][0]))

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
                orig_logits = m_output_orig(m_norm_orig(orig_batch.to(device)))
                quant_logits = m_output_quant(m_norm_quant(quant_batch.to(device)))
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
