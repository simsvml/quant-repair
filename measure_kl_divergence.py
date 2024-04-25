import os
import re
import sys
from tempfile import TemporaryDirectory
import torch
from torch import Tensor
from torchtune.modules import quantized
from torchtune.models import convert_weights
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer_transformers
from torchtune.utils import FullModelHFCheckpointer, set_default_dtype


ADAPTER_FILE_NAME_REGEX = re.compile(r'adapter_([0-9]+)\.pt')

LAYER_PARAMETERS = {
    'attn.k_proj.weight',
    'attn.output_proj.weight',
    'attn.q_proj.weight',
    'attn.v_proj.weight',
    'mlp.w1.weight',
    'mlp.w2.weight',
    'mlp.w3.weight',
    'mlp_norm.scale',
    'sa_norm.scale',
}


class SlowTransformerDecoder(torch.nn.Module):
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
        for key in LAYER_PARAMETERS:
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
    next_token_logits = logits[0, -1, :]
    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
    top_k_probs = torch.nn.functional.softmax(top_k_logits / temperature, dim=-1)
    print(next_token_logits.shape)
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
    dir_path = sys.argv[1]

    device = torch.device('cuda')


    print('load original model from %r' % (dir_path,))

    # The checkpointer always writes a config to its output directory.  We send
    # that to a temporary directory that will be deleted automatically on exit.
    with TemporaryDirectory() as checkpointer_output_dir:
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=dir_path,
            checkpoint_files=[f for f in os.listdir(dir_path) if f.endswith('.safetensors')],
            model_type='LLAMA3',
            output_dir=checkpointer_output_dir,
        )
        checkpoint_dict = checkpointer.load_checkpoint()
        del checkpointer
    #print(sorted(checkpoint_dict.keys()))
    #print(sorted(checkpoint_dict['model'].keys()))


    print('build model')

    model_orig = SlowTransformerDecoder(llama3_8b(), checkpoint_dict['model'])
    model_orig = model_orig.to(device)

    num_layers = 32


    # Load alternate weights
    print('load alternate weights')

    alt_state_dict = checkpoint_dict['model'].copy()
    for layer_index in range(num_layers):
        layer_dir = os.path.join(dir_path, 'output', str(layer_index))
        if not os.path.isdir(layer_dir):
            continue
        max_adapter_index = None
        for f in os.listdir(layer_dir):
            m = ADAPTER_FILE_NAME_REGEX.match(f)
            if not m:
                continue
            idx = int(m.group(1))
            if max_adapter_index is None or idx > max_adapter_index:
                max_adapter_index = idx
        if max_adapter_index is None:
            continue
        layer_path = os.path.join(layer_dir, 'adapter_%d.pt' % max_adapter_index)
        print('  ' + layer_path)
        layer_dict = torch.load(layer_path, weights_only=True, map_location='cpu')
        print('layer', layer_index, sorted(layer_dict.keys()))
        for key, value in layer_dict.items():
            assert key in LAYER_PARAMETERS
            alt_state_dict['layers.%d.%s' % (layer_index, key)] = value


    print('build alternate model')

    from torchtune.utils._checkpointing._checkpointer_utils import load_gguf
    gguf_dict = load_gguf(
        os.path.join(dir_path, 'ggml-model-Q6_K.gguf'),
        filter_name_prefix = 'dont_load_any_tensors',
    )
    quant_map = convert_weights.gguf_to_tune(gguf_dict['gguf_quant_map'])
    #print(quant_map)

    model_alt_base = llama3_8b()
    quantized.replace_modules(model_alt_base, quant_map)
    model_alt = SlowTransformerDecoder(model_alt_base, alt_state_dict)
    model_alt = model_alt.to(device)


    print('load tokenizer')
    tokenizer = llama3_tokenizer_transformers(os.path.join(dir_path, 'tokenizer.json'))


    print('test run')
    tokens = tokenizer.encode('Hello, my name', add_eos=False)
            #'<|start_header_id|>system<|end_header_id|>\n\n'
            #'You are a helpful chatbot assistant.<|eot_id|>'
            #'<|start_header_id|>user<|end_header_id|>\n\n'
            #'Hello, my name is John.<|eot_id|>'
            #'<|start_header_id|>assistant<|end_header_id|>\n\n',
    print(tokens)

    x_orig = model_orig.forward(torch.tensor([tokens], device=device, dtype=torch.int))
    print(x_orig[0, -1])
    print(top_tokens(tokenizer, x_orig, top_k=10))

    x_alt = model_alt.forward(torch.tensor([tokens], device=device, dtype=torch.int))
    print(x_alt[0, -1])
    print(top_tokens(tokenizer, x_alt, top_k=10))


    y_orig = torch.nn.functional.log_softmax(x_orig, dim=-1)
    y_alt = torch.nn.functional.log_softmax(x_alt, dim=-1)

    kl_div_fn = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
    kl_div = kl_div_fn(y_alt, y_orig)
    print(kl_div)



    #loss_fn = torch.nn.MSELoss()
    #loss = loss_fn(x_orig, x_alt)
    #print(loss)

#
#    model_orig2 = llama3_8b().to(device)
#    model_orig2.load_state_dict(checkpoint_dict['model'])
#    print('test run2')
#    print(model_orig2.forward(torch.tensor([[65, 66, 67]], device=device,
#                                          dtype=torch.int)))


    # Create model

    #model_orig = llama3_8b()


    # Load alternate layer weights

    



if __name__ == '__main__':
    main()
