from functools import partial
from typing import Optional, Any, Dict, Tuple, List, Mapping, Callable
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchtune.data import sharegpt_to_llama2_messages
from torchtune.modules.tokenizers import Tokenizer


class DatasetAdapter(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]

    def map(self, func) -> 'MapDatasetAdapter':
        return MapDatasetAdapter(self, func)

    def shuffle(self, seed: int = 0) -> 'ShuffleDatasetAdapter':
        return ShuffleDatasetAdapter(self, seed)

    def skip(self, n: int) -> 'SkipDatasetAdapter':
        return SkipDatasetAdapter(self, n)

    def take(self, n: int) -> 'TakeDatasetAdapter':
        return TakeDatasetAdapter(self, n)

    def reversed(self) -> 'ReversedDatasetAdapter':
        return ReversedDatasetAdapter(self)

class MapDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        func: Callable,
    ):
        super().__init__(dataset)
        self._func = func

    def __getitem__(self, index: int) -> Any:
        return self._func(super().__getitem__(index))

class ShuffleDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
    ):
        super().__init__(dataset)
        g = torch.Generator()
        g.manual_seed(seed)
        self._order = torch.randperm(len(dataset), generator = g, device = 'cpu').tolist()

    def __getitem__(self, index: int) -> Any:
        shuffled_index = self._order[index]
        return super().__getitem__(shuffled_index)

class SkipDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        skip: int,
    ):
        super().__init__(dataset)
        self._skip = skip

    def __len__(self) -> int:
        return max(0, super().__len__() - self._skip)

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index + self._skip)

class TakeDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        take: int,
    ):
        super().__init__(dataset)
        self._take = take

    def __len__(self) -> int:
        return min(super().__len__(), self._take)

class ReversedDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
    ):
        super().__init__(dataset)

    def __getitem__(self, index: int) -> Any:
        rev_index = super().__len__() - 1 - index
        return super().__getitem__(rev_index)


def make_tokenize_func(
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> Callable[[str], List[int]]:
    def tokenize_func(text: str) -> List[int]:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
        tokens = tokens[:max_seq_len]
        return torch.tensor(tokens, dtype = torch.int32, device = 'cpu')
    return tokenize_func

def make_sharegpt_chat_tokenize_func(
    tokenizer: Tokenizer,
    max_seq_len: int,
    train_on_input: bool = True,
) -> Callable[[str], List[int]]:
    def sharegpt_chat_tokenize_func(x: dict) -> List[int]:
        messages = sharegpt_to_llama2_messages(x, train_on_input)
        tokens, mask = tokenizer.tokenize_messages(messages, max_seq_len)
        tokens = tokens[:max_seq_len]
        return torch.tensor(tokens, dtype = torch.int32, device = 'cpu')
    return sharegpt_chat_tokenize_func

def load_slimorca_dataset(
    tokenizer: Tokenizer,
    max_seq_len: Optional[int] = None,
    # TODO: `train_on_input` currently has no effect.  Fix or remove it.
    train_on_input: bool = False,
    split: str = 'train',
    **kwargs,
) -> DatasetAdapter:
    return MapDatasetAdapter(
        load_dataset('Open-Orca/SlimOrca-Dedup', split = split, **kwargs),
        make_sharegpt_chat_tokenize_func(tokenizer, max_seq_len, train_on_input),
    )

def load_wikitext_dataset(
    tokenizer: Tokenizer,
    max_seq_len: Optional[int] = None,
    name: str = 'wikitext-2-raw-v1',
    split: str = 'test',
    **kwargs,
) -> DatasetAdapter:
    return MapDatasetAdapter(
        load_dataset('wikitext', name = name, split = split, **kwargs),
        make_tokenize_func(tokenizer, max_seq_len),
    )
