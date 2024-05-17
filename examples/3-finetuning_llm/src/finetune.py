# taken and adapted from https://github.com/brevdev/notebooks/blob/main/llama3_finetune_inference.ipynb

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

base_model_id = "meta-llama/Meta-Llama-3-8B"
dataset_name = "scooterman/guanaco-llama3-1k"
new_model = "hpc_lt3_tutorial-llama3-8b-SFT"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Output directory where the results and checkpoint are stored
output_dir = "./results"

# Number of training epochs - how many times does the model see the whole dataset
num_train_epochs = 1  # Increase this for a larger finetune

# Enable fp16/bf16 training. This is the type of each weight. Since we are on an A100
# we can set bf16 to true because it can handle that type of computation
bf16 = True

# Batch size is the number of training examples used to train a single forward and backward pass.
per_device_train_batch_size = 4

# Gradients are accumulated over multiple mini-batches before updating the model weights.
# This allows for effectively training with a larger batch size on hardware with limited memory
gradient_accumulation_steps = 2

# memory optimization technique that reduces RAM usage during training by intermittently storing
# intermediate activations instead of retaining them throughout the entire forward pass, trading
# computational time for lower memory consumption.
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Number of training steps (overrides num_train_epochs)
max_steps = 5

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 100

# Log every X updates steps
logging_steps = 5

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
