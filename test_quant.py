import os
import sys
from tempfile import TemporaryDirectory
import torch
from torch import nn
from torch import Tensor
from torchtune.models import convert_weights
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer_transformers
from torchtune.modules import quantized
from torchtune.utils import FullModelGGUFCheckpointer, set_default_dtype
from gguf import GGUFReader, GGMLQuantizationType
from torchtune.utils import gguf_quant


class SlowTransformerDecoder(nn.Module):
    '''Like `TransformerDecoder`, but loads the weights from CPU to GPU between
    layers.  As a result, it uses very little GPU memory, but runs slower.'''
    def __init__(self, orig_model, state_dict):
        super().__init__()

        self.tok_embeddings = orig_model.tok_embeddings
        self.layer = orig_model.layers[0]
        self.num_layers = len(orig_model.layers)
        self.norm = orig_model.norm
        self.output = orig_model.output

        self.full_state_dict = state_dict

        self.tok_embeddings.load_state_dict({
            'weight': state_dict['tok_embeddings.weight'],
        })
        self.norm.load_state_dict({
            'scale': state_dict['norm.scale'],
        })
        self.output.load_state_dict({
            'weight': state_dict['output.weight'],
        })

    def layer_state_dict(self, i):
        state_dict = {}
        for key, _ in self.layer.named_parameters():
            state_dict[key] = self.full_state_dict['layers.%d.%s' % (i, key)]
        return state_dict

    def forward(self, tokens: Tensor) -> Tensor:
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        # Attention mask and input pos are always `None`.
        mask = None
        input_pos = None

        for i in range(self.num_layers):
            # Update `self.layer` with the weights for this layer
            self.layer.load_state_dict(self.layer_state_dict(i))

            # shape: [b, s, d]
            h = self.layer(h, mask, input_pos)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = self.output(h).float()
        return output


def top_tokens(tokenizer, logits, top_k=3, temperature=1.0):
    # https://medium.com/@pashashaik/natural-language-generation-from-scratch-in-large-language-models-with-pytorch-4d9379635316
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    top_k_probs = torch.nn.functional.softmax(top_k_logits / temperature, dim=-1)
    print(logits.shape)
    print(top_k_logits)
    print(top_k_indices)
    print(top_k_probs)
    return [(prob.item(), tokenizer.decode(token)) for prob, token in zip(top_k_probs, top_k_indices)]


def main():
    with set_default_dtype(torch.bfloat16):
        with torch.no_grad():
            run()

def run():
    assert len(sys.argv) == 2
    gguf_path = sys.argv[1]
    dir_path = os.path.dirname(gguf_path)

    device = torch.device('cuda')

    print('load weights from %r' % (gguf_path,))

    reader = GGUFReader(gguf_path)

    state_dict = {}
    quant_map = {}
    for tensor in reader.tensors:
        quant = GGMLQuantizationType(tensor.tensor_type)
        quant_map[tensor.name] = quant

        shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
            default=len(tensor.shape))
        shape = tuple(reversed(tensor.shape[:shape_length]))

        print(quant.name, shape, tensor.name)

        if quant in quantized.UNQUANTIZED_TYPES:
            state_dict[tensor.name] = torch.from_numpy(tensor.data).view(*shape)
        elif quant == GGMLQuantizationType.Q6_K:
            qs, scales, d = gguf_quant.unpack_q6_k(tensor.data)

            data2 = gguf_quant.pack_q6_k(qs, scales, d)
            if (data2 != tensor.data).any():
                raise AssertionError('bad pack/unpack of %s tensor %s' % (
                    quant, tensor.name))

            if not tensor.name.startswith('blk.'):
                # `quantized.replace_modules` doesn't yet support these.
                data = gguf_quant.dequant_q6_k(qs, scales, d)
                state_dict[tensor.name] = torch.from_numpy(data).view(*shape)
                continue

            state_dict.update({
                '%s_quant.k_qs' % tensor.name: torch.from_numpy(qs),
                '%s_quant.k_scales' % tensor.name: torch.from_numpy(scales),
                '%s_quant.k_d' % tensor.name: torch.from_numpy(d),
            })
        elif quant == GGMLQuantizationType.Q5_K:
            x = gguf_quant.unpack_q5_k(tensor.data)
            raise AssertionError('quant %s not implemented for tensor %s' % (
                quant, tensor.name))
        else:
            raise AssertionError('quant %s not implemented for tensor %s' % (
                quant, tensor.name))

    quant_map = convert_weights.gguf_to_tune(quant_map)
    state_dict = convert_weights.gguf_to_tune(state_dict)

    print('build model')
    model = llama3_8b()
    model = SlowTransformerDecoder(model, state_dict)
    layer_quant_map = {name: quant_map['layers.0.%s' % name]
        for name, _ in model.layer.named_parameters()}
    quantized.replace_modules(model.layer, layer_quant_map)
    model = model.to(device)
    #model.load_state_dict(state_dict)

    #print(model.layers[0].attn.k_proj.weight_quant.forward())

    print('load tokenizer')
    tokenizer = llama3_tokenizer_transformers(os.path.join(dir_path, 'tokenizer.json'))

    print('test run')
    tokens = tokenizer.encode('Hello, my name is', add_eos=False)
    print(tokens)

    x = model.forward(torch.tensor([tokens], device=device, dtype=torch.int))
    print(x[0, -1])
    print(top_tokens(tokenizer, x[0, -2], top_k=10))
    print(top_tokens(tokenizer, x[0, -1], top_k=10))


if __name__ == '__main__':
    main()
