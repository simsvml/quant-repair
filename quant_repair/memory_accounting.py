from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Iterable
import weakref
import torch
from torch import Tensor


DISABLE_MEMORY_ACCOUNTING = False

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
        self.log_file = None

    @staticmethod
    def disable():
        global DISABLE_MEMORY_ACCOUNTING
        DISABLE_MEMORY_ACCOUNTING = True

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

        if self.log_file is None:
            self.log_file = open('memory_%s.log' % time.strftime('%Y%m%d_%H%M%S'), 'w')

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


        def print_(s):
            print(s)
            print(s, file = self.log_file, flush = True)


        print_(' === Memory Report ===')
        if header is not None:
            print_(header)

        for device_name, device_sizes in sorted(sizes.items()):
            print_('\nMemory usage for device %s:' % device_name)
            device_total = 0
            for desc, size in sorted(device_sizes.items()):
                print_('  %7.3f GB   %s' % (size / 1024**3, desc))
                device_total += size
            print_('  %7.3f GB   Total' % (device_total / 1024**3))
            if device_name.startswith('cuda'):
                torch_size = torch.cuda.memory_allocated(device_name)
                print_('  %7.3f GB   Total (pytorch reported)' % (torch_size / 1024**3))
                delta = torch_size - device_total
                print_('  %7.3f GB   Unaccounted' % (delta / 1024**3))

MEMORY_ACCOUNTING = MemoryAccounting()
