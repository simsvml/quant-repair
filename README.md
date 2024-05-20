This is a fork of [torchtune](https://github.com/pytorch/torchtune) with
various hacks and experiments related to improving performance of quantized
models.

# Usage

This code is not in a usable state.  The precise usage of the scripts is not
documented.  Here are some rough guidelines:

* Unpack your quantized GGUF model, quantize, and convert to pytorch format
  using `quant_unpack_dequant.py ggml-model-Qx.gguf out_quant_dir/`.
* Train a LoRA using `train_repair_lora_streaming.py orig_dir/ quant_dir/`.
  `orig_dir` should have the original FP16 model in safetensors format.
  Training settings are hardcoded in various places in the script.
* Evaluate using `measure_kl_divergence.py orig_dir/ quant_dir/
  [quant_dir/lora.pt]`.  The last argument is optional; run without it to get
  the baseline KL-div and with it to get the KL-div with the LoRA applied.
