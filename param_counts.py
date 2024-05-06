from quant_repair.architecture import Llama3Arch


def run(arch):
    head_dim = arch.embed_dim // arch.num_heads
    shapes = [
        # layer.attn.q_proj
        (arch.embed_dim, arch.num_heads * head_dim),
        # layer.attn.k_proj
        (arch.embed_dim, arch.num_kv_heads * head_dim),
        # layer.attn.v_proj
        (arch.embed_dim, arch.num_kv_heads * head_dim),
        # layer.attn.output_proj
        (arch.embed_dim, arch.embed_dim),
        # mlp.w1
        (arch.embed_dim, arch.intermediate_dim),
        # mlp.w2
        (arch.intermediate_dim, arch.embed_dim),
        # mlp.w3
        (arch.embed_dim, arch.intermediate_dim),
    ]
    layer_lora_params = lambda r: sum(n * r + m * r for n,m in shapes)
    layer_total_params = sum(n*m for n,m in shapes)

    input_output_params = arch.vocab_size * arch.embed_dim
    input_output_lora_params = lambda r: arch.vocab_size * r + r * arch.embed_dim

    model_total_params = 2 * input_output_params + arch.num_layers * layer_total_params
    model_lora_params = lambda r: \
        2 * input_output_lora_params(r) + arch.num_layers * layer_lora_params(r)

    # TODO: This omit the various norm layers
    print('total params: %d' % model_total_params)
    for r in (16, 32, 64):
        print('total lora (rank %d) params: %d' % (r, model_lora_params(r)))
        print('lora (rank %d, fp16) bpw equivalent: %.3f' %
              (r, model_lora_params(r) * 16 / model_total_params))
    print()
    print('forward layer param counts:')
    print('  %13d  tok_embeddings' % input_output_params)
    print('  %13d  layer' % layer_total_params)
    print('  %13d  output' % input_output_params)
    print('  %13d  total' % (input_output_params * 2 + layer_total_params))

    print()
    print('forward pass vram usage: %.1f GB' %
          ((input_output_params * 2 + layer_total_params) * 2 / 1024**3))
    print('training module vram usage:')
    # 2 bytes per value, 1 weight + 1 gradient value per model weight
    print('  %13.1f MB  tok_embeddings' % (input_output_params * 2 * 2 / 1024**2))
    print('  %13.1f MB  layer' % (layer_total_params * 2 * 2 / 1024**2))
    print('  %13.1f MB  output' % (input_output_params * 2 * 2 / 1024**2))
    print('  %13.1f MB  5 layers + output' %
          ((5 * layer_total_params + input_output_params) * 2 * 2 / 1024**2))

print('\n\nLlama3 8B:')
run(Llama3Arch.llama3_8b())

print('\n\nLlama3 70B:')
run(Llama3Arch.llama3_70b())
