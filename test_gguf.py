import os
import sys
from tempfile import TemporaryDirectory
import torch
from torch import nn
from torchtune.models import convert_weights
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer_transformers
from torchtune.modules import quantized
from torchtune.utils import FullModelGGUFCheckpointer, set_default_dtype
from gguf import GGMLQuantizationType


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


    print('load original model from %r' % (gguf_path,))

    if os.path.exists('test_gguf_checkpoint.pt'):
        checkpoint_dict = torch.load('test_gguf_checkpoint.pt')
    else:
        # The checkpointer always writes a config to its output directory.  We send
        # that to a temporary directory that will be deleted automatically on exit.
        with TemporaryDirectory() as checkpointer_output_dir:
            checkpointer = FullModelGGUFCheckpointer(
                checkpoint_dir=os.path.dirname(gguf_path),
                checkpoint_files=[os.path.basename(gguf_path)],
                model_type='LLAMA3',
                output_dir=checkpointer_output_dir,
            )
            checkpoint_dict = checkpointer.load_checkpoint()
            del checkpointer
        print(sorted(checkpoint_dict.keys()))
        print(sorted(checkpoint_dict['model'].keys()))

        torch.save(checkpoint_dict, 'test_gguf_checkpoint.pt')

    quant_map = checkpoint_dict['model'].pop('gguf_quant_map')
    quant_map = convert_weights.gguf_to_tune(quant_map)
    print(quant_map)

    print(sorted(checkpoint_dict['model'].keys()))

    #for k,v in checkpoint_dict['model'].items():
    #    print(k, v.shape)

    print('build model')
    model = llama3_8b()
    quantized.replace_modules(model, quant_map)
    model = model.to(device)
    model.load_state_dict(checkpoint_dict['model'])

    print(model.layers[0].attn.k_proj.weight_quant.forward())

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
