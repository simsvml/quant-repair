from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Optional, Dict, Iterable
from .memory_accounting import MEMORY_ACCOUNTING


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

