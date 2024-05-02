from functools import partial
from typing import Optional, Any, Dict, Tuple, List, Mapping
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchtune.datasets import slimorca_dataset
from torchtune.modules.tokenizers import Tokenizer
from torchtune.utils import padded_collate


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

def build_data_loader(
    dataset: Dataset,
    tokenizer: Tokenizer,
    batch_size: int,
    seed: int = 0,
) -> Tuple[DistributedSampler, DataLoader]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=seed,
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
    return sampler, dataloader

def load_slimorca_dataset(
    tokenizer: Tokenizer,
    max_seq_len: Optional[int] = None,
    train_on_input: bool = False,
    **kwargs,
) -> Tuple[DistributedSampler, DataLoader]:
    dataset = slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
    )
    return build_data_loader(dataset, tokenizer, **kwargs)

def load_wikitext_dataset(
    tokenizer: Tokenizer,
    max_seq_len: Optional[int] = None,
    **kwargs,
) -> Tuple[DistributedSampler, DataLoader]:
    dataset = TextDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        source='wikitext',
        name='wikitext-2-raw-v1',
        split='test',
    )
    return build_data_loader(dataset, tokenizer, **kwargs)
