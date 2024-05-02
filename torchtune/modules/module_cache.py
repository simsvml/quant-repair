from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from torch import Tensor
from torch import nn
from typing import Optional, Any, Dict

from torchtune.modules import quantized
from torchtune.modules.attention import CausalSelfAttention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.rms_norm import RMSNorm
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings
from torchtune.modules.transformer import TransformerDecoderLayer


@dataclass
class ModuleCacheEntry:
    module: Optional[nn.Module]
    weights_desc: Any

class ModuleCache:
    def __init__(self, cache_size=4):
        @lru_cache(maxsize=cache_size)
        def cache(key: Any) -> ModuleCacheEntry:
            return ModuleCacheEntry(None, None)

        # `self._cache(key)` will return an existing entry if there is one, and
        # otherwise will create and return a new, blank entry.
        self._cache = cache

    def get_module(
        self,
        key: Any,
        weights_desc: Any,
        module_func: Callable[[Any], nn.Module],
        weights_func: Callable[[Any, Any], Dict[str, Tensor]],
    ) -> nn.Module:
        """
        Get the module identified by `key`, with its weights initialized
        according to `weights_desc`.  If the desired module is not present in
        the cache, it will be created by calling `module_func(key)`.  Then, if
        the module's weights are not initialized to `weights_desc`, they will
        be initialized by calling `weights_func(key, weights_desc)` to obtain a
        state dict for the module.
        """
        entry = self._cache(key)
        need_weights = False
        if entry.module is None:
            entry.module = module_func(key)
            # Always initialize weights, even if the caller passes `None` for
            # `weights_desc`, which happens to equal the default value for
            # newly created entries.
            need_weights = True
        if need_weights or entry.weights_desc != weights_desc:
            state_dict = weights_func(key, weights_desc)
            entry.module.load_state_dict(state_dict)
            entry.weights_desc = weights_desc
        return entry.module

from quant_repair.architecture import Llama3Arch
