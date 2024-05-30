from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
import math
import os
from pprint import pprint
import sys
import time
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Set, Any, Union, Iterable, Callable
import weakref
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchtune.models.llama3 import llama3_tokenizer_transformers
from torchtune.modules import quantized, lr_schedulers
from torchtune.modules import RotaryPositionalEmbeddings
from torchtune.utils import set_default_dtype
from gguf import GGMLQuantizationType
from quant_repair.architecture import Llama3Arch
from quant_repair.common import build_module, load_weights, init_lora_weights
from quant_repair.datasets import load_slimorca_dataset, load_wikitext_dataset
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks
from quant_repair import functional as QRF
from quant_repair.memory_accounting import MEMORY_ACCOUNTING
from quant_repair import model_util as QRM
from quant_repair.model_util.misc import weights_getter
from quant_repair.model_util.llama3_lora import TrainableParams, LayerTrainableParams
from quant_repair.model_util.superbatch import run_forward_superbatch2
from quant_repair.modules import LowRankAdapter, QuantLowRankAdapter, WithAdapter
from quant_repair.offload import TensorOffload
import quant_repair.loader
from quant_repair.loader import StateDictLoader, SafetensorsLoader, GGUFLoader


def main():
    with set_default_dtype(torch.bfloat16):
        #with torch.no_grad():
            run()

def run():
    assert len(sys.argv) in (3, 4)
    orig_path = sys.argv[1]
    quant_path = sys.argv[2]
    if len(sys.argv) >= 4:
        lora_path = sys.argv[3]
    else:
        lora_path = None

    MEMORY_ACCOUNTING.disable()

    device = torch.device('cuda')
    arch = Llama3Arch.llama3_70b()

    print('loading original weights from %r' % (orig_path,))
    orig_loader = SafetensorsLoader(orig_path,
        convert = quant_repair.loader.hf_conversion(arch))
    assert orig_loader.has('layers.%d.attn.q_proj.weight' % (arch.num_layers - 1)), \
            'architecture mismatch: %r is missing layers' % orig_path
    assert not orig_loader.has('layers.%d.attn.q_proj.weight' % arch.num_layers), \
            'architecture mismatch: %r has too many layers' % orig_path

    print('loading quantized weights from %r' % (quant_path,))
    quant_loader = GGUFLoader(quant_path,
        convert = quant_repair.loader.llama_cpp_conversion())
    assert quant_loader.has('layers.%d.attn.q_proj.weight' % (arch.num_layers - 1)), \
            'architecture mismatch: %r is missing layers' % quant_path
    assert not quant_loader.has('layers.%d.attn.q_proj.weight' % arch.num_layers), \
            'architecture mismatch: %r has too many layers' % quant_path

    if lora_path is not None:
        print('loading trained lora from %r' % (lora_path,))
        lora_state_dict = torch.load(lora_path)
        if 'params' in lora_state_dict:
            lora_state_dict = lora_state_dict['params']
        lora_loader = StateDictLoader(lora_state_dict)
        lora_params = QRM.llama3_lora.load_trainable_params(
            lora_loader, arch.num_layers, device)
    else:
        lora_params = None

    if os.path.isdir(orig_path):
        orig_dir = orig_path
    else:
        orig_dir = os.path.dirname(orig_path)
    tokenizer_json_path = os.path.join(orig_path, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    # Build llama3 model
    model_common = QRM.llama3.ModelCommon(
        rope = RotaryPositionalEmbeddings(
            dim = arch.head_dim(),
            max_seq_len = arch.max_seq_len,
            base = arch.rope_base,
        ).to(device),
    )

    model = QRM.llama3.make_model(arch, model_common)

    def embedding_with_lora():
        base = QRF.Embedding()
        adapter = QRF.EmbeddingLowRankAdapter()
        return QRF.WithAdapter(base, adapter)

    def linear_with_lora():
        base = QRF.Linear()
        adapter = QRF.LowRankAdapter()
        return QRF.WithAdapter(base, adapter)

    model_with_lora = QRM.llama3.make_model(
        arch, model_common, linear_with_lora, embedding_with_lora)


    # Training config
    max_seq_len = 1024
    batch_size = 1
    total_epochs = 1
    #max_steps_per_epoch = 0
    max_steps_per_epoch = 4000
    #max_steps_per_epoch = 9000
    #max_steps_per_epoch = 2500
    #max_steps_per_epoch = 1000
    #max_steps_per_epoch = 2500
    gradient_accumulation_steps = 16

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    #superbatch_mem_gb = 32
    superbatch_mem_gb = 8


    # Test config
    test_steps = 500


    # Set up dataset and loader
    print('loading dataset')
    sampler, dataloader = load_slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
        seed=0,
        batch_size=batch_size,
    )
    #sampler, dataloader = load_wikitext_dataset(
    #    tokenizer = tokenizer,
    #    max_seq_len = max_seq_len,
    #    seed = 0,
    #    batch_size = batch_size,
    #)

    # Loss function
    loss_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')

    MEMORY_ACCOUNTING.report('before testing loop')

    # Testing loop

    # Match indentation of train_repair_lora_streaming.py
    if True:
        sampler.set_epoch(0)

        samples_iter = (tokens for tokens, labels in dataloader)
        print('skipping %d samples' % max_samples)
        samples_iter = itertools.islice(samples_iter, max_samples, max_samples + test_steps)

        embeds_orig = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb / 2)
        embeds_quant = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb / 2)
        superbatch_limit = embeds_orig.tokens_free()


        pbar_orig = tqdm(desc='orig', total=test_steps)
        pbar_quant = tqdm(desc='quant', total=test_steps)
        pbar_loss = tqdm(desc='loss', total=test_steps)
        pbar_superbatch_forward = tqdm(desc='superbatch forward', total=2 + arch.num_layers)
        pbar_superbatch_layer = tqdm(desc='superbatch layer', total=1)

        loss_sum = 0.0
        loss_count = 0
        total_tokens = 0

        for superbatch_samples in sized_chunks(samples_iter, superbatch_limit,
                lambda t: t.numel()):
            run_forward_superbatch2(
                superbatch_samples,
                embeds_orig,
                arch.num_layers,
                lambda: QRM.llama3.build_forward_tok_embeddings(model, orig_loader, device),
                lambda i: QRM.llama3.build_forward_layer(model, orig_loader, i, device),
                lambda: QRM.llama3.build_forward_norm(model, orig_loader, device),
                device,
                pbar_forward = pbar_superbatch_forward,
                pbar_layer = pbar_superbatch_layer,
            )
            pbar_orig.update(len(superbatch_samples))

            MEMORY_ACCOUNTING.report('after orig superbatch')

            if lora_params is None:
                run_forward_superbatch2(
                    superbatch_samples,
                    embeds_quant,
                    arch.num_layers,
                    lambda: QRM.llama3.build_forward_tok_embeddings(model, quant_loader, device),
                    lambda i: QRM.llama3.build_forward_layer(model, quant_loader, i, device),
                    lambda: QRM.llama3.build_forward_norm(model, quant_loader, device),
                    device,
                    pbar_forward = pbar_superbatch_forward,
                    pbar_layer = pbar_superbatch_layer,
                )
                pbar_quant.update(len(superbatch_samples))

                m_quant = QRM.llama3.build_forward_output(model, quant_loader, device)
            else:
                run_forward_superbatch2(
                    superbatch_samples,
                    embeds_quant,
                    arch.num_layers,
                    lambda: QRM.llama3_lora.build_trainable_tok_embeddings(
                        model_with_lora, quant_loader, lora_params, device),
                    lambda i: QRM.llama3_lora.build_trainable_layer(
                        model_with_lora, quant_loader, lora_params, i, device),
                    lambda: QRM.llama3_lora.build_trainable_norm(
                        model_with_lora, quant_loader, lora_params, device),
                    device,
                    pbar_forward = pbar_superbatch_forward,
                    pbar_layer = pbar_superbatch_layer,
                )
                pbar_quant.update(len(superbatch_samples))

                m_quant = QRM.llama3_lora.build_trainable_output(
                    model_with_lora, quant_loader, lora_params, device)

            MEMORY_ACCOUNTING.report('after quant superbatch')

            m_orig = QRM.llama3.build_forward_output(model, orig_loader, device)

            for i in range(len(superbatch_samples)):
                orig_logits = m_orig(embeds_orig[i].to(device))
                MEMORY_ACCOUNTING.register(orig_logits, 'orig_logits')
                orig_log_prob = F.log_softmax(orig_logits, dim=-1)
                MEMORY_ACCOUNTING.register(orig_log_prob, 'orig_log_prob')
                orig_log_prob = orig_log_prob.view(-1, arch.vocab_size)
                del orig_logits

                quant_logits = m_quant(embeds_quant[i].to(device))
                MEMORY_ACCOUNTING.register(quant_logits, 'quant_logits')
                quant_log_prob = F.log_softmax(quant_logits, dim=-1)
                MEMORY_ACCOUNTING.register(quant_log_prob, 'quant_log_prob')
                quant_log_prob = quant_log_prob.view(-1, arch.vocab_size)
                del quant_logits

                loss = loss_fn(quant_log_prob, orig_log_prob)
                MEMORY_ACCOUNTING.register(loss, 'loss')

                loss_sum += loss.item()
                loss_count += 1
                total_tokens += superbatch_samples[i].numel()

                del orig_log_prob, quant_log_prob, loss

                pbar_loss.update()

        pbar_orig.close()
        pbar_quant.close()
        pbar_loss.close()
        pbar_superbatch_forward.close()
        pbar_superbatch_layer.close()

        print('\n\n%d total tokens' % total_tokens)
        print('loss = %.6e' % (loss_sum / loss_count))

    MEMORY_ACCOUNTING.report('after training loop')


if __name__ == '__main__':
    main()
