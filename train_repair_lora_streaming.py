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
from quant_repair.datasets import load_slimorca_dataset
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks
from quant_repair import functional as QRF
from quant_repair.modules import LowRankAdapter, QuantLowRankAdapter, WithAdapter
from quant_repair.weights import load_weights_safetensors_hf, \
    CheckpointStateDict, QuantizedCheckpointLoader


def module_index_to_name(arch, index):
    if index == 0:
        return 'tok_embeddings'
    else:
        index -= 1
    if index < arch.num_layers:
        return 'layer%d' % index
    else:
        index -= arch.num_layers
    assert index == 0
    return 'norm_output'


DISABLE_MEMORY_ACCOUNTING = True

@dataclass(frozen=True)
class MemoryAccountingEntry:
    tensor: weakref.ref
    size: int
    device: torch.device
    desc: str

class MemoryAccounting:
    def __init__(self):
        # Map from `id(tensor)` to a `MemoryAccountingEntry`.  Using
        # `id(tensor)` as the key ensures we don't record the same tensor
        # twice.
        self.entries = {}

    def register(self, tensor, desc):
        if DISABLE_MEMORY_ACCOUNTING:
            return

        if tensor.grad is not None:
            self.register(tensor.grad, 'gradient for ' + desc)

        key = id(tensor)

        old_entry = self.entries.get(key)
        if old_entry is not None and old_entry.tensor() is tensor:
            # Don't replace the original entry.
            return

        self.entries[key] = MemoryAccountingEntry(
            tensor = weakref.ref(tensor),
            size = tensor.nbytes,
            device = tensor.device,
            desc = desc,
        )

    def register_all(self, tensors: Iterable[Tensor], desc):
        if DISABLE_MEMORY_ACCOUNTING:
            return

        for tensor in tensors:
            self.register(tensor, desc)

    def register_params(self, params, desc):
        if DISABLE_MEMORY_ACCOUNTING:
            return

        for tensor in params.tensors():
            self.register(tensor, desc)

    def report(self, header=None):
        if DISABLE_MEMORY_ACCOUNTING:
            return

        # Bring the `entries` set up to date by removing stale items and adding
        # missing gradient tensors.
        del_keys = []
        add_grads = []
        for key, entry in self.entries.items():
            tensor = entry.tensor()
            if tensor is None:
                # Weak ref has expired - the tensor has been deallocated.
                del_keys.append(key)
                continue

            if tensor.grad is not None:
                add_grads.append((tensor.grad, entry.desc))

        for key in del_keys:
            if self.entries[key].tensor() is None:
                del self.entries[key]

        for (grad_tensor, desc) in add_grads:
            grad_key = id(grad_tensor)
            if grad_key not in self.entries or self.entries[grad_key].tensor() is None:
                self.register(grad_tensor, 'gradient (late) for ' + desc)

        # `sizes[device][desc]` is the total size in bytes of all tensors
        # matching `device` and `desc`.
        sizes = defaultdict(lambda: defaultdict(float))
        for entry in self.entries.values():
            sizes[str(entry.device)][entry.desc] += entry.size


        print(' === Memory Report ===')
        if header is not None:
            print(header)

        for device_name, device_sizes in sorted(sizes.items()):
            print('\nMemory usage for device %s:' % device_name)
            device_total = 0
            for desc, size in sorted(device_sizes.items()):
                print('  %7.3f GB   %s' % (size / 1024**3, desc))
                device_total += size
            print('  %7.3f GB   Total' % (device_total / 1024**3))
            if device_name.startswith('cuda'):
                torch_size = torch.cuda.memory_allocated(device_name)
                print('  %7.3f GB   Total (pytorch reported)' % (torch_size / 1024**3))
                delta = torch_size - device_total
                print('  %7.3f GB   Unaccounted' % (delta / 1024**3))

MEMORY_ACCOUNTING = MemoryAccounting()


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
    def get1(key):
        return loader.get(key)[key].to(device)

    embeds.clear()
    if pbar_forward is not None:
        pbar_forward.reset(2 + len(model.layers))

    # tok_embeddings
    m = build_forward_tok_embeddings(model, loader, device)
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
        m = build_forward_layer(model, loader, i, device)
        embeds.apply(m, device = device, pbar = pbar_layer)
        del m
        if pbar_forward is not None:
            pbar_forward.update()

    # norm
    m = build_forward_norm(model, loader, device)
    embeds.apply(m, device = device, pbar = pbar_layer)
    del m
    if pbar_forward is not None:
        pbar_forward.update()


def run_backward_step(
    x: Tensor,
    grad_y: Tensor,
    f,
) -> Tensor:
    """
    Given input `x` and output gradient `grad_y`, where `y = f(x)`, compute the
    input gradient `grad_x`.
    """
    x.requires_grad_(True)
    MEMORY_ACCOUNTING.register(x, 'run_backward_step x')
    MEMORY_ACCOUNTING.register(grad_y, 'run_backward_step grad_y')
    y = f(x)
    MEMORY_ACCOUNTING.register(y, 'run_backward_step y')
    y.backward(grad_y)
    MEMORY_ACCOUNTING.register(x.grad, 'run_backward_step x.grad')
    return x.grad

def run_initial_backward_step(
    x: Tensor,
    f,
) -> Tuple[Tensor, Tensor]:
    """
    Given input `x`, compute `y = f(x)` and the gradient of `x`.
    """
    x.requires_grad_(True)
    MEMORY_ACCOUNTING.register(x, 'run_initial_backward_step x')
    y = f(x)
    MEMORY_ACCOUNTING.register(y, 'run_initial_backward_step y')
    y.backward()
    MEMORY_ACCOUNTING.register(x.grad, 'run_initial_backward_step x.grad')
    return y, x.grad

def run_final_backward_step(
    x: Tensor,
    grad_y: Tensor,
    f,
):
    """
    Given input `x` and output gradient `grad_y`, where `y = f(x)`, update
    parameter graidents.  The gradient of `x` is not returned.
    """
    MEMORY_ACCOUNTING.register(x, 'run_initial_backward_step x')
    MEMORY_ACCOUNTING.register(grad_y, 'run_backward_step grad_y')
    y = f(x)
    MEMORY_ACCOUNTING.register(y, 'run_backward_step y')
    y.backward(grad_y)
    return x.grad


@dataclass
class TensorOffloadEntry:
    tensor: Tensor
    tensor_gpu: Optional[Tensor]
    name: str
    group: str
    partition: int
    read_only: bool

class TensorOffload:
    """
    Helper for managing tensor offloading from GPU to CPU.

    Initialization has three phases:

    1. Define the group names and ordering by calling `add_group(group_name)`.
    2. Add tensors by calling `add_tensor(group_name, tensor_name, tensor)`.
       Tensors should be on the CPU at this point.
    3. Call `build_partitions()`.  This will partition the groups into chunks
       of size at most `vram_limit_gb`.

    After initializing, retrieve tensors by calling `get(tensor_name)`.  When a
    tensor is retrieved, all tensors in its partition (which at minimum
    includes all tensors in the same group) will be moved to VRAM.
    """
    def __init__(self, device, vram_limit_gb):
        self.device = device
        self.vram_limit = int(vram_limit_gb * 1024 ** 3)
        self.tensors = {}
        self.groups = []

        # For each partition, this has a list of tensors in that partition.
        self.partition_tensors = []
        self.current_partition = None

    def add_group(self, group_name):
        self.groups.append(group_name)

    def add_tensor(self, group_name, tensor_name, tensor, read_only=False):
        assert tensor_name not in self.tensors, 'duplicate tensor %r' % (tensor_name,)
        self.tensors[tensor_name] = TensorOffloadEntry(
            tensor = tensor.to('cpu'),
            tensor_gpu = None,
            name = tensor_name,
            group = group_name,
            partition = -1,
            read_only = read_only,
        )

    def build_partitions(self):
        group_tensors = {}
        group_sizes = {}

        # Initialize keys of `group_tensors`, and also check for duplicate
        # groups.
        for group in self.groups:
            assert group not in group_tensors, 'duplicate group %r' % (group,)
            group_tensors[group] = []
            group_sizes[group] = 0

        for entry in self.tensors.values():
            group_tensors[entry.group].append(entry)
            group_sizes[entry.group] += entry.tensor.nbytes

        partition_groups = []
        current_groups = []
        current_size = 0
        for group in self.groups:
            group_size = group_sizes[group]
            if current_size + group_size > self.vram_limit:
                # Finish the current group.
                if len(current_groups) > 0:
                    partition_groups.append(current_groups)
                    current_groups = []
                    current_size = 0
            if group_size > self.vram_limit:
                print('warning: group %s exceeds limit: %d (%.3f GB) > %d (%.3f GB)' %
                    (group, group_size, group_size / 1024**3,
                        self.vram_limit, self.vram_limit / 1024**3))
                # Oversized groups get a partition to themselves.
                assert len(current_groups) == 0
            current_groups.append(group)
            current_size += group_size
        if len(current_groups) > 0:
            partition_groups.append(current_groups)

        for i, groups in enumerate(partition_groups):
            print('partition %d: %s' % (i, groups))
            current_tensors = []
            for group in groups:
                for tensor in group_tensors[group]:
                    tensor.partition = i
                    current_tensors.append(tensor)
            self.partition_tensors.append(current_tensors)

    def has(self, key: str) -> bool:
        return key in self.tensors

    def _load_partition(self, partition):
        if partition == self.current_partition:
            return

        #tqdm.write('load_partition: %s -> %s' % (self.current_partition, partition))

        if self.current_partition is not None:
            # Unload the current partition to free up space.
            for entry in self.partition_tensors[self.current_partition]:
                if not entry.read_only:
                    entry.tensor.copy_(entry.tensor_gpu, non_blocking=True)
                del entry.tensor_gpu

        # Wait for copy_ operations to finish so the old GPU tensors can be
        # discarded before we start allocating new ones.
        torch.cuda.synchronize()

        for entry in self.partition_tensors[partition]:
            entry.tensor_gpu = entry.tensor.to(self.device)
            MEMORY_ACCOUNTING.register(entry.tensor_gpu, 'offload group %s' % entry.group)

        self.current_partition = partition

    def get(
        self,
        key: str,
        result_key: Optional[str] = None,
        dequant: bool = False,
    ) -> Dict[str, Tensor]:
        if result_key is None:
            result_key = key

        partition = self.tensors[key].partition
        if partition != self.current_partition:
            self._load_partition(partition)
        assert self.tensors[key].tensor_gpu is not None
        return {result_key: self.tensors[key].tensor_gpu}

    def get_multi(
        self,
        keys: Iterable[str],
        result_keys: Optional[str] = None,
        dequant: bool = False,
    ) -> Dict[str, Tensor]:
        if result_keys is not None:
            keys_iter = zip(keys, result_keys, strict=True)
        else:
            keys_iter = ((k, k) for k in keys)

        result = {}
        first_partition = None
        for key, result_key in keys_iter:
            partition = self.tensors[key].partition
            if first_partition is None:
                first_partition = partition
            else:
                if partition != first_partition:
                    # We could optimize this case, but it's unlikely to occur.
                    print('warning: get_multi spans multiple partitions; this will be slow')
            if partition != self.current_partition:
                self._load_partition(partition)
            assert self.tensors[key].tensor_gpu is not None
            result[result_key] = self.tensors[key].tensor_gpu

        return result


@contextmanager
def record_time(dest, key):
    start = time.time()
    yield
    end = time.time()
    dest[key] += end - start

def main():
    with set_default_dtype(torch.bfloat16):
        #with torch.no_grad():
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


    # Build llama3 model
    rope = RotaryPositionalEmbeddings(
        dim = arch.head_dim(),
        max_seq_len = arch.max_seq_len,
        base = arch.rope_base,
    ).to(device)
    model = QRF.TransformerDecoder(
        tok_embeddings = QRF.Embedding(),
        layers = [
            QRF.TransformerDecoderLayer(
                attn = QRF.CausalSelfAttention(
                    embed_dim = arch.embed_dim,
                    num_heads = arch.num_heads,
                    num_kv_heads = arch.num_kv_heads,
                    head_dim = arch.head_dim(),
                    q_proj = QRF.Linear(),
                    k_proj = QRF.Linear(),
                    v_proj = QRF.Linear(),
                    output_proj = QRF.Linear(),
                    pos_embeddings = rope,
                ),
                mlp = QRF.FeedForward(
                    gate_proj = QRF.Linear(),
                    down_proj = QRF.Linear(),
                    up_proj = QRF.Linear(),
                    activation = nn.SiLU(),
                ),
                sa_norm = QRF.RMSNorm(eps = arch.norm_eps),
                mlp_norm = QRF.RMSNorm(eps = arch.norm_eps),
            ) for i in range(arch.num_layers)
        ],
        norm = QRF.RMSNorm(eps = arch.norm_eps),
        output = QRF.Linear(),
    )

    def embedding_with_lora():
        base = QRF.Embedding()
        adapter = QRF.EmbeddingLowRankAdapter()
        return QRF.WithAdapter(base, adapter)

    def linear_with_lora():
        base = QRF.Linear()
        adapter = QRF.LowRankAdapter()
        return QRF.WithAdapter(base, adapter)

    model_with_lora = QRF.TransformerDecoder(
        tok_embeddings = embedding_with_lora(),
        layers = [
            QRF.TransformerDecoderLayer(
                attn = QRF.CausalSelfAttention(
                    embed_dim = arch.embed_dim,
                    num_heads = arch.num_heads,
                    num_kv_heads = arch.num_kv_heads,
                    head_dim = arch.head_dim(),
                    q_proj = linear_with_lora(),
                    k_proj = linear_with_lora(),
                    v_proj = linear_with_lora(),
                    output_proj = linear_with_lora(),
                    pos_embeddings = rope,
                ),
                mlp = QRF.FeedForward(
                    gate_proj = linear_with_lora(),
                    down_proj = linear_with_lora(),
                    up_proj = linear_with_lora(),
                    activation = nn.SiLU(),
                ),
                sa_norm = QRF.RMSNorm(eps = arch.norm_eps),
                mlp_norm = QRF.RMSNorm(eps = arch.norm_eps),
            ) for i in range(arch.num_layers)
        ],
        norm = QRF.RMSNorm(eps = arch.norm_eps),
        output = linear_with_lora(),
    )


    print('initializing trainable parameters')

    # Norm scales are initialized from the base model's weights.  LoRA
    # parameters are initialized to default values.

    lora_rank = 32

    def init_lora_params(
        in_features: int,
        out_features: int,
        rank: int = lora_rank,
        # TODO: Try float16 for easier llama.cpp compat
        dtype: torch.dtype = torch.bfloat16,
    ) -> QRF.LowRankAdapterParams:
        return QRF.LowRankAdapterParams(
            lora_a = torch.empty((rank, in_features), dtype=dtype, device=device).normal_(),
            lora_b = torch.zeros((out_features, rank), dtype=dtype, device=device),
        )

    def init_layer_params(layer_index: int) -> LayerTrainableParams:
        embed_dim = arch.embed_dim
        num_heads = arch.num_heads
        num_kv_heads = arch.num_kv_heads
        head_dim = arch.head_dim()
        hidden_dim = arch.hidden_dim()

        get1 = weights_getter(quant_weights, device)

        return LayerTrainableParams(
            q_proj = init_lora_params(embed_dim, num_heads * head_dim),
            k_proj = init_lora_params(embed_dim, num_kv_heads * head_dim),
            v_proj = init_lora_params(embed_dim, num_kv_heads * head_dim),
            output_proj = init_lora_params(embed_dim, embed_dim),
            gate_proj = init_lora_params(embed_dim, hidden_dim),
            down_proj = init_lora_params(hidden_dim, embed_dim),
            up_proj = init_lora_params(embed_dim, hidden_dim),
            sa_norm = QRF.RMSNormParams(
                get1('layers.%d.sa_norm.scale' % layer_index).to(torch.bfloat16)),
            mlp_norm = QRF.RMSNormParams(
                get1('layers.%d.mlp_norm.scale' % layer_index).to(torch.bfloat16)),
        )

    def init_train_params() -> TrainableParams:
        get1 = weights_getter(quant_weights, device)
        return TrainableParams(
            tok_embeddings = init_lora_params(arch.vocab_size, arch.embed_dim),
            layers = [init_layer_params(i) for i in range(arch.num_layers)],
            norm = QRF.RMSNormParams(get1('norm.scale').to(torch.bfloat16)),
            output = init_lora_params(arch.embed_dim, arch.vocab_size),
        )

    train_params = init_train_params()
    MEMORY_ACCOUNTING.register_params(train_params, 'trainable params')

    for tensor in train_params.tensors():
        tensor.requires_grad_(True)


    # Training config
    max_seq_len = 1024
    batch_size = 1
    total_epochs = 1
    max_steps_per_epoch = 4
    #max_steps_per_epoch = 2500
    #max_steps_per_epoch = 1000
    #max_steps_per_epoch = 2500
    gradient_accumulation_steps = 8

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    #superbatch_mem_gb = 32
    superbatch_mem_gb = 8

    # Set up dataset and loader
    print('loading dataset')
    sampler, dataloader = load_slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
        seed=0,
        batch_size=batch_size,
    )

    # Optimizer, learning rate schedule, and loss function

#    optimizer = torch.optim.AdamW(
#        list(train_params.tensors()),
#        lr = 2.5e-6,
#    )
#    lr_scheduler = lr_schedulers.get_exponential_schedule(
#        optimizer,
#        start_factor = 1.0,
#        end_factor = 0.1,
#        num_training_steps = total_epochs * max_steps_per_epoch,
#    )

    optimizer = torch.optim.AdamW(
        list(train_params.tensors()),
        lr = 7e-6,
        weight_decay = 0.01,
    )
    lr_scheduler = lr_schedulers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 100,
        num_training_steps = total_epochs * max_steps_per_epoch,
    )

    loss_fn = nn.MSELoss()
    kl_div_loss_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')


    # Offload setup
    print('loading quantized weights for offload')
    offload = TensorOffload(device, vram_limit_gb=8)

    offload.add_group('tok_embeddings')
    for i in range(arch.num_layers):
        offload.add_group('layers.%d' % i)
    offload.add_group('norm')
    offload.add_group('output')

    def offload_init_tensors():
        get1 = weights_getter(quant_weights, 'cpu')
        def add_weight(group, name):
            full_name = '%s.%s' % (group, name)
            offload.add_tensor(group, full_name, get1(full_name), read_only=True)

        add_weight('tok_embeddings', 'weight')
        for i in range(arch.num_layers):
            add_weight('layers.%d' % i, 'attn.q_proj.weight')
            add_weight('layers.%d' % i, 'attn.k_proj.weight')
            add_weight('layers.%d' % i, 'attn.v_proj.weight')
            add_weight('layers.%d' % i, 'attn.output_proj.weight')
            add_weight('layers.%d' % i, 'mlp.w1.weight')
            add_weight('layers.%d' % i, 'mlp.w2.weight')
            add_weight('layers.%d' % i, 'mlp.w3.weight')
            add_weight('layers.%d' % i, 'sa_norm.scale')
            add_weight('layers.%d' % i, 'mlp_norm.scale')
        add_weight('norm', 'scale')
        add_weight('output', 'weight')
    offload_init_tensors()

    offload.build_partitions()

    # Use `offload` for fetching base model weights.
    quant_weights = offload


    MEMORY_ACCOUNTING.report('before training loop')


    # Training loop
    print('training %d parameter tensors' % (len(list(train_params.tensors()))))
    log_file = open('train_%d.log' % time.time(), 'w')
    metrics = {
        'quant_time': 0.,
        'orig_time': 0.,
        'train_time': 0.,
        'opt_time': 0.,
        'loss': None,
        'lr': None,
        'gpu_resources': None,
    }
    for curr_epoch in range(total_epochs):
        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        sampler.set_epoch(curr_epoch)

        samples_iter = (tokens for tokens, labels in dataloader)
        samples_iter = itertools.islice(samples_iter, max_samples)

        embeds_orig = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb)
        superbatch_limit = embeds_orig.tokens_free()


        pbar_samples = tqdm(desc='samples', total=max_samples, smoothing=0)
        pbar_superbatch = tqdm(desc='superbatch', total=max_samples)
        pbar_superbatch_forward = tqdm(desc='superbatch forward', total=2 + arch.num_layers)
        pbar_superbatch_layer = tqdm(desc='superbatch layer', total=1)
        pbar_train_forward = tqdm(desc='train forward', total=1)
        pbar_train_backward = tqdm(desc='train backward', total=1)

        for superbatch_samples in sized_chunks(samples_iter, superbatch_limit,
                lambda t: t.numel()):
            run_forward_superbatch(
                model,
                orig_weights,
                superbatch_samples,
                embeds_orig,
                device,
                pbar_forward = pbar_superbatch_forward,
                pbar_layer = pbar_superbatch_layer,
            )
            pbar_superbatch.update(len(superbatch_samples))

            MEMORY_ACCOUNTING.report('after forward superbatch')

            # Train using the collected embeddings.
            with record_time(metrics, 'train_time'):
                for samples_start in range(0, len(superbatch_samples), gradient_accumulation_steps):
                    samples = superbatch_samples[samples_start :
                        samples_start + gradient_accumulation_steps]
                    samples = [t.to(device).requires_grad_(False) for t in samples]
                    MEMORY_ACCOUNTING.register_all(samples, 'samples')

                    pbar_train_forward.reset(2 + arch.num_layers)
                    pbar_train_backward.reset(1 + 2 + arch.num_layers)

                    activation_checkpoints = []
                    with torch.no_grad():
                        m = build_trainable_tok_embeddings(
                            model_with_lora, quant_weights, train_params, device)
                        activations = [m(sample) for sample in samples]
                        MEMORY_ACCOUNTING.register_all(activations, 'activations')
                        activation_checkpoints.append(activations)
                        pbar_train_forward.update()
                        del m

                        for i in range(arch.num_layers):
                            m = build_trainable_layer(
                                model_with_lora, quant_weights, train_params, i, device)
                            activations = [m(act) for act in activations]
                            MEMORY_ACCOUNTING.register(activations, 'activations')
                            activation_checkpoints.append(activations)
                            pbar_train_forward.update()
                            del m

                        m = build_trainable_norm(
                            model_with_lora, quant_weights, train_params, device)
                        activations = [m(act) for act in activations]
                        MEMORY_ACCOUNTING.register_all(activations, 'activations')
                        # Norm output is not recorded as a checkpoint.
                        pbar_train_forward.update()
                        del m

                        # Final activations are kept around for later use in
                        # calc_loss.

                    MEMORY_ACCOUNTING.report('after train forward pass')

                    # Run loss forward and backward
                    m_orig = build_forward_output(model, orig_weights, device)
                    m_train = build_trainable_output(
                        model_with_lora, quant_weights, train_params, device)

                    # Sum of loss values (used only for logging).
                    loss_sum = 0.0
                    gradients = []
                    for i in range(len(samples)):
                        def calc_loss(train_embeds: Tensor) -> Tensor:
                            orig_logits = m_orig(embeds_orig[samples_start + i].to(device))
                            MEMORY_ACCOUNTING.register(orig_logits, 'orig_logits')
                            orig_log_prob = F.log_softmax(orig_logits, dim=-1)
                            MEMORY_ACCOUNTING.register(orig_log_prob, 'orig_log_prob')
                            orig_log_prob = orig_log_prob.view(-1, arch.vocab_size)
                            del orig_logits

                            train_logits = m_train(train_embeds)
                            MEMORY_ACCOUNTING.register(train_logits, 'train_logits')
                            train_log_prob = F.log_softmax(train_logits, dim=-1)
                            MEMORY_ACCOUNTING.register(train_log_prob, 'train_log_prob')
                            train_log_prob = train_log_prob.view(-1, arch.vocab_size)
                            del train_logits

                            loss = kl_div_loss_fn(train_log_prob, orig_log_prob)
                            MEMORY_ACCOUNTING.register(loss, 'loss')
                            return loss

                        train_embeds = activations[i]
                        loss, grad = run_initial_backward_step(train_embeds, calc_loss)
                        MEMORY_ACCOUNTING.register(grad, 'backward gradients')

                        gradients.append(grad)
                        loss_sum += loss.item()

                    loss_avg = loss_sum / len(samples)
                    del m_orig, m_train
                    pbar_train_backward.update()

                    # Run remaining backward steps.

                    m = build_trainable_norm(
                        model_with_lora, quant_weights, train_params, device)
                    for i, act in enumerate(activation_checkpoints.pop()):
                        gradients[i] = run_backward_step(act, gradients[i], m)
                    MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
                    del m, act
                    pbar_train_backward.update()

                    for i in reversed(range(arch.num_layers)):
                        m = build_trainable_layer(
                            model_with_lora, quant_weights, train_params, i, device)
                        for j, act in enumerate(activation_checkpoints.pop()):
                            gradients[j] = run_backward_step(act, gradients[j], m)
                        MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
                        del m, act
                        pbar_train_backward.update()

                    m = build_trainable_tok_embeddings(
                        model_with_lora, quant_weights, train_params, device)
                    for i, sample in enumerate(samples):
                        run_final_backward_step(sample, gradients[i], m)
                    del m, sample, gradients
                    pbar_train_backward.update()

                    MEMORY_ACCOUNTING.report('after train backward pass')

                    with record_time(metrics, 'opt_time'):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()

                    pbar_samples.set_description(
                        f"Loss: {loss_avg:.6e}"
                    )
                    pbar_samples.update(len(samples))

                    metrics['loss'] = loss_avg
                    metrics['lr'] = optimizer.param_groups[0]["lr"]
                    metrics['gpu_resources'] = torch.cuda.memory_allocated()
                    json.dump(metrics, log_file)
                    log_file.write('\n')
                    log_file.flush()

        pbar_samples.close()
        pbar_superbatch.close()
        pbar_superbatch_forward.close()
        pbar_superbatch_layer.close()
        pbar_train_forward.close()
        pbar_train_backward.close()

    MEMORY_ACCOUNTING.report('after training loop')


    state_dict = {}

    def save_low_rank_adapter_params(name, params):
        state_dict[name + '.lora_a'] = params.lora_a
        state_dict[name + '.lora_b'] = params.lora_b

    def save_rms_norm_params(name ,params):
        state_dict[name + '.scale'] = params.scale

    def save_layer_trainable_params(name, params):
        save_low_rank_adapter_params(name + '.q_proj', params.q_proj)
        save_low_rank_adapter_params(name + '.k_proj', params.k_proj)
        save_low_rank_adapter_params(name + '.v_proj', params.v_proj)
        save_low_rank_adapter_params(name + '.output_proj', params.output_proj)
        save_low_rank_adapter_params(name + '.gate_proj', params.gate_proj)
        save_low_rank_adapter_params(name + '.down_proj', params.down_proj)
        save_low_rank_adapter_params(name + '.up_proj', params.up_proj)
        save_rms_norm_params(name + '.sa_norm', params.sa_norm)
        save_rms_norm_params(name + '.mlp_norm', params.mlp_norm)

    def save_trainable_params(name, params):
        save_low_rank_adapter_params(name + '.tok_embeddings', params.tok_embeddings)
        for i, layer_params in enumerate(params.layers):
            save_layer_trainable_params('%s.layers.%d' % (name, i), layer_params)
        save_rms_norm_params(name + '.norm', params.norm)
        save_low_rank_adapter_params(name + '.output', params.output)

    save_trainable_params('params', train_params)
    checkpoint_path = os.path.join(quant_dir, 'repair_ckpt.pt')
    torch.save(state_dict, checkpoint_path)
    print('\n\nsaved %s' % checkpoint_path)


if __name__ == '__main__':
    main()
