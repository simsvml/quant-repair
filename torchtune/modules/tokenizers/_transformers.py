# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from sentencepiece import SentencePieceProcessor
from torchtune.data._types import Message
from torchtune.data._utils import truncate
from transformers import PreTrainedTokenizerFast

class TransformersTokenizer:
    """A wrapper around PreTrainedTokenizerFast.

    Args:
        path (str): Path to pretrained tokenizer JSON file.
    """

    def __init__(
        self,
        path: str,
    ):
        self._inner = PreTrainedTokenizerFast(tokenizer_file=path)

        def special_token(text: str) -> int:
            tokens = self._inner.encode(text, add_special_tokens=False)
            assert len(tokens) == 1
            token = tokens[0]
            assert isinstance(token, int), 'bad token %r for %r' % (token, text)
            return token

        self.bos_id = special_token("<|begin_of_text|>")
        self.eos_id = special_token("<|end_of_text|>")
        self.pad_id = 0
        self.start_header_id = special_token("<|start_header_id|>")
        self.end_header_id = special_token("<|end_header_id|>")
        self.eot_id = special_token("<|eot_id|>")
        #self.eom_id = special_token("<|eom_id|>")
        #self.python_tag = special_token("<|python_tag|>")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
        Returns:
            List[int]: The encoded token IDs.
        """
        tokens = self._inner.encode(text, add_special_tokens=False)
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return self._inner.decode(ids)

    def tokenize_message(
        self, message: Message, tokenize_header: bool = False
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            List[int]: The list of token ids.
        """
        if tokenize_header:
            tokenized_header = (
                [self.start_header_id]
                + self.encode(message.role.strip(), add_bos=False, add_eos=False)
                + [self.end_header_id]
                + self.encode("\n\n", add_bos=False, add_eos=False)
            )
        else:
            tokenized_header = []
        tokenized_body = self.encode(
            message.content.strip(), add_bos=False, add_eos=False
        )
        if message.ipython:
            #tokenized_body = [self.python_tag] + tokenized_body
            raise ValueError('ipython messages are not supported (no python_tag available)')
        tokenized_message = tokenized_header + tokenized_body
        if message.eot:
            tokenized_message = tokenized_message + [self.eot_id]
        else:
            #tokenized_message = tokenized_message + [self.eom_id]
            raise ValueError('messages must be EOT (no eom_id available)')
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        max_seq_len: Optional[int] = None,
        tokenize_header: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            max_seq_len (Optional[int]): The maximum sequence length.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]
        for message in messages:
            tokenized_message = self.tokenize_message(
                message, tokenize_header=tokenize_header
            )
            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if max_seq_len and len(tokens) >= max_seq_len:
                break
        tokens = tokens + [self.eos_id]
        mask = mask + [True]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True)
        assert None not in tokens, 'bad tokens %r for %r' % (tokens, messages)
        return tokens, mask
