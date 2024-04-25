# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.models import convert_weights
from torchtune.modules import quantized
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils._checkpointing._checkpointer_utils import load_gguf
from tqdm import tqdm

log = utils.get_logger("DEBUG")

import torch._dynamo
torch._dynamo.config.suppress_errors = True


class QuantizeLayerBitnetRecipeSingleDevice(FTRecipeInterface):
    """
    Recipe for quantizing one layer of a model to Bitnet and then repairing it
    with post-training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.

    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if (
            self._dtype == torch.bfloat16
            and self._device != torch.device("cpu")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Full bf16 training is not supported on this hardware.")
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.log_every_n_steps if cfg.log_every_n_steps else 1
        self._log_peak_memory_every_n_steps = 100

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.total_training_steps = 0

        self._target_layer = cfg.target_layer

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

    def load_checkpoint(
        self,
        cfg_checkpointer_orig: DictConfig,
        cfg_checkpointer_layer: DictConfig,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer_orig = config.instantiate(
            cfg_checkpointer_orig,
            resume_from_checkpoint=False,
        )
        self._checkpointer_layer = config.instantiate(
            cfg_checkpointer_layer,
            resume_from_checkpoint=self._resume_from_checkpoint,
            load_single_layer=self._target_layer,
        )
        orig_checkpoint_dict = self._checkpointer_orig.load_checkpoint()
        layer_checkpoint_dict = self._checkpointer_layer.load_checkpoint()

        if self._resume_from_checkpoint:
            if utils.ADAPTER_KEY not in layer_checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(layer_checkpoint_dict)
        return orig_checkpoint_dict, layer_checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._model_compile = cfg.compile
        orig_checkpoint_dict, layer_checkpoint_dict = self.load_checkpoint(
            cfg_checkpointer_orig=cfg.checkpointer_orig,
            cfg_checkpointer_layer=cfg.checkpointer_layer,
        )

        self._model_orig, self._model_layer = self._setup_model(
            cfg_model_orig=cfg.model_orig,
            cfg_model_layer=cfg.model_layer,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=cfg.compile,
            orig_model_state_dict=orig_checkpoint_dict[utils.MODEL_KEY],
            layer_model_state_dict=layer_checkpoint_dict[utils.MODEL_KEY],
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None
            ),
        )

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.total_training_steps = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.total_training_steps - 1,
        )

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)

    def _setup_model(
        self,
        cfg_model_orig: DictConfig,
        cfg_model_layer: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        orig_model_state_dict: Dict[str, Any],
        layer_model_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with utils.set_default_dtype(torch.bfloat16), self._device:
            model_orig = config.instantiate(
                cfg_model_orig,
                target_layer=self._target_layer,
            )
        with utils.set_default_dtype(self._dtype), self._device:
            model_layer = config.instantiate(cfg_model_layer)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model_layer, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        # Set up the original model
        orig_missing, orig_unexpected = model_orig.load_state_dict(
            orig_model_state_dict, strict=False
        )
        assert not orig_missing, 'missing parameters: %r' % (orig_missing,)
        # TODO: Check that only the expected keys are present in `orig_unexpected`

        # Set up the layer model.  We must convert the layer names to be
        # appropriate for a single layer instead of a full model.

        quant_map = layer_model_state_dict.pop('gguf_quant_map')
        #print(quant_map)
        sub_quant_map = {}
        for name, param in model_layer.named_parameters():
            full_name = 'layers.%d.%s' % (self._target_layer, name)
            sub_quant_map[name] = quant_map[full_name]
        # `quantized.replace_modules` uses `get_default_dtype` internally, so
        # set the default before calling.
        with utils.set_default_dtype(self._dtype), self._device:
            quantized.replace_modules(model_layer, sub_quant_map)

        sub_state_dict = {}
        for name, param in model_layer.named_parameters():
            full_name = 'layers.%d.%s' % (self._target_layer, name)
            sub_state_dict[name] = layer_model_state_dict[full_name]

        layer_missing, layer_unexpected = model_layer.load_state_dict(
            sub_state_dict, strict=False
        )
        assert not layer_missing, 'missing parameters for layer: %r' % (layer_missing,)
        assert not layer_unexpected, 'unexpected parameters for layer: %r' % (layer_unexpected,)

        # Validate model params were loaded in with the expected dtype
        #utils.validate_expected_param_dtype(
        #    model_orig.named_parameters(), dtype=self._dtype
        #)

        model_orig.requires_grad_(False)
        model_layer.requires_grad_(True)

        log.info(f"Model is initialized with precision {self._dtype}.")
        # Compile model, if enabled.
        if compile_model:
            log.info("Compiling model with torch.compile...")
            model_orig = utils.wrap_compile(model_orig)
            model_layer = utils.wrap_compile(model_layer)
        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)
        return model_orig, model_layer

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model_layer.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        ds = config.instantiate(
            cfg_dataset,
            tokenizer=self._tokenizer,
        )
        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                #ignore_idx=self._loss_fn.ignore_index,
                ignore_idx=-100,
            ),
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.OPT_KEY: self._optimizer.state_dict(),
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        state_dict = self._model_layer.state_dict()
        renamed_state_dict = {'layers.%d.%s' % (self._target_layer, key): value
            for key, value in state_dict.items()}

        ckpt_dict = {
                utils.MODEL_KEY: renamed_state_dict,
                }
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
            ckpt_dict[utils.OPT_KEY] = self._optimizer.state_dict()
            #if not self._optimizer_in_bwd:
            #    ckpt_dict[utils.OPT_KEY] = self._optimizer.state_dict()
            #else:
            #    ckpt_dict[utils.OPT_KEY] = self._optim_ckpt_wrapper.state_dict()

        self._checkpointer_layer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def train(self) -> None:
        """
        The core training loop.
        """

        if self._model_compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        torch.autograd.set_detect_anomaly(True)
        def nan_hook(self, inp, output):
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

        #for submodule in self._model_layer.modules():
        #    submodule.register_forward_hook(nan_hook)

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            loss_acc = 0

            # Optionally profile the training loop
            with self._profiler:
                for idx, batch in enumerate(pbar := tqdm(self._dataloader)):
                    if (
                        self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps)
                        == self.max_steps_per_epoch
                    ):
                        break

                    if self._profiler_enabled:
                        self._profiler.step()

                    input_ids, labels = batch
                    input_ids = input_ids.to(self._device)
                    labels = labels.to(self._device)

                    input_embd, output_embd = self._model_orig(input_ids)
                    input_embd = input_embd.to(self._dtype)
                    output_embd = output_embd.to(self._dtype)
                    layer_output_embd = self._model_layer(input_embd)

                    output_embd_flat = output_embd.flatten(0, 1)
                    layer_output_embd_flat = layer_output_embd.flatten(0, 1)
                    assert output_embd_flat.shape == \
                            (output_embd.shape[0] * output_embd.shape[1],
                             output_embd.shape[2])

                    #loss = self._loss_fn(output_embd_flat, layer_output_embd_flat,
                        #torch.ones((output_embd_flat.shape[0],), device=self._device))
                    #print('output', output_embd)
                    #print('layer output', layer_output_embd)
                    loss = self._loss_fn(output_embd_flat, layer_output_embd_flat)

                    if self.total_training_steps % self._log_every_n_steps == 0:
                        pbar.set_description(
                            f"{curr_epoch+1}|{idx+1}|Loss: {loss.item()}"
                        )
                        self._metric_logger.log_dict(
                            {
                                "loss": loss.item(),
                                "lr": self._optimizer.param_groups[0]["lr"],
                                "gpu_resources": torch.cuda.memory_allocated(),
                            },
                            step=self.total_training_steps,  # Each step is unique, not limited to each epoch
                        )
                    #print(loss)
                    loss = loss / self._gradient_accumulation_steps
                    loss.backward()
                    #def check(name, t):
                    #    assert not torch.isnan(t).any(), 'found nan in %s gradient' % name
                    #    print(name, t)
                    #check('w2 qs', self._model_layer.mlp.w2.weight_quant.k_qs.grad)
                    #print(self._model_layer.mlp.w2.weight_quant.k_scales.grad)
                    #print(self._model_layer.mlp.w2.weight_quant.k_d.grad)
                    #check('w1 qs', self._model_layer.mlp.w1.weight_quant.k_qs.grad)
                    #print(self._model_layer.mlp.w1.weight_quant.k_scales.grad)
                    #print(self._model_layer.mlp.w1.weight_quant.k_d.grad)
                    #check('mlp_norm scale', self._model_layer.mlp_norm.scale.grad)
                    #check('output qs', self._model_layer.attn.output_proj.weight_quant.k_qs.grad)
                    #check('v qs', self._model_layer.attn.v_proj.weight_quant.k_qs.grad)
                    #check('k qs', self._model_layer.attn.k_proj.weight_quant.k_qs.grad)
                    #check('q qs', self._model_layer.attn.q_proj.weight_quant.k_qs.grad)
                    #check('sa_norm scale', self._model_layer.sa_norm.scale.grad)

                    loss_acc += float(loss)

                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        if isinstance(self._lr_scheduler, ReduceLROnPlateau):
                            self._lr_scheduler.step(loss_acc)
                            loss_acc = 0
                        else:
                            self._lr_scheduler.step()
                        # Update the number of steps when the weights are updated
                        self.total_training_steps += 1
                    # Log peak memory for iteration
                    if (
                        self.total_training_steps % self._log_peak_memory_every_n_steps
                        == 0
                        and self._device.type == "cuda"
                    ):
                        # Log peak memory for iteration
                        memory_stats = utils.get_memory_stats(device=self._device)
                        self._metric_logger.log_dict(
                            memory_stats, step=self.total_training_steps
                        )
            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="QuantizeLayerBitnetRecipeSingleDevice", cfg=cfg)
    recipe = QuantizeLayerBitnetRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
