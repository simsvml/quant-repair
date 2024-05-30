This is a fork of [torchtune](https://github.com/pytorch/torchtune) with
various hacks and experiments related to improving performance of quantized
models.

# Usage

1.  Write a training config file.  See `l3_70b_example.toml` for an example.

    To use the example config as-is, create a symlink `models/70b_orig`
    pointing to a Llama 3 checkout (containing safetensors files and
    `tokenizer.json`) and create a symlink `models/70b_quant.gguf` pointing to
    a quantized GGUF of the same model.

2.  Use the config to create an initial checkpoint:

    ```sh
    python3 train_repair_lora2.py init config.toml checkpoint.pt
    ```

    The config settings will be copied into the checkpoint file, so editing the
    config file after this will have no effect on the training.  If you need to
    adjust the configuration of an existing checkpoint, see the section below.

3.  Train:

    ```sh
    python3 train_repair_lora2.py train checkpoint.pt
    ```

    This will periodically overwrite `checkpoint.pt` with an updated training
    checkpoint.  You can stop the training at any time with `^C` and resume it
    from the latest checkpoint by running the same command again.


## Reconfiguring an existing checkpoint

The training configuration is stored inside the checkpoint file.  It's possible
to change this embedded config, which may be useful if you moved the base model
or checkpoint and need to update the paths.  However, the training script
doesn't have any special logic to account for config changes, so adjusting
things like learning rate schedule or total tokens in the middle of a training
run might produce strange results.

To change the config of an existing checkpoint:

```sh
python3 train_repair_lora2.py extract_config checkpoint.pt >checkpoint_config.toml
# Edit checkpoint_config.toml
python3 train_repair_lora2.py update_config checkpoint.pt checkpoint_config.toml
# Answer prompts to confirm that you want to change the settings
```
