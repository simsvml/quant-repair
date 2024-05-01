"""
Forward pass implementation.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable

@dataclass(frozen=True)
class BatchInfo:
    shape: torch.Size
    start: int

class SuperbatchEmbeddings:
    def __init__(self, arch, *, ram_gb=1):
        ram_bytes = int(ram_gb * 1024**3)
        bytes_per_token = arch.embed_dim * torch.get_default_dtype().itemsize
        self._max_tokens = ram_bytes // bytes_per_token

        self._embed_dim = arch.embed_dim
        self._batches = []
        self._tokens_used = 0
        self._buffer = torch.empty((self._max_tokens, self._embed_dim), device='cpu')

    def clear(self):
        self._batches = []
        self._tokens_used = 0

    def tokens_free(self) -> int:
        return self._max_tokens - self._tokens_used

    def __len__(self) -> int:
        return len(self._batches)

    def append(self, batch: Tensor):
        assert batch.shape[-1] == self._embed_dim
        shape = batch.shape[:-1]
        new_tokens = shape.numel()
        assert new_tokens <= self.tokens_free()

        start = self._tokens_used
        end = start + new_tokens
        self._buffer[start:end, :] = batch.view(new_tokens, self._embed_dim)
        self._tokens_used = end
        self._batches.append(BatchInfo(shape, start))

    def __getitem__(self, i: int) -> Tensor:
        """
        Get the current embeddings for batch `i`.
        """
        info = self._batches[i]
        start = info.start
        end = start + info.shape.numel()
        batch = self._buffer[start:end, :]
        return batch.view(info.shape + (self._embed_dim,))

    def __iter__(self):
        for info in self._batches:
            start = info.start
            end = start + info.shape.numel()
            batch = self._buffer[start:end, :]
            yield batch.view(info.shape + (self._embed_dim,))

    def apply(self, m: Callable[[Tensor], Tensor], device=None, tqdm_kwargs={}):
        """
        Update each batch by applying `m` to it.
        """
        for batch in tqdm(self, total=len(self), **tqdm_kwargs):
            new = m(batch.to(device))
            batch.copy_(new)

def sized_chunks(xs, limit, measure):
    """
    Divide `xs` into chunks whose size according to `measure` does not exceed
    `limit`.
    """
    buf = []
    free = limit
    for x in xs:
        size = measure(x)
        if size > free:
            yield buf
            buf = []
            free = limit
        buf.append(x)
        # If `measure(x)` exceeds `limit`, this will send `free` negative and
        # cause `x` to be emitted in a separate chunk on its own.
        free -= size
    if len(buf) > 0:
        yield buf

