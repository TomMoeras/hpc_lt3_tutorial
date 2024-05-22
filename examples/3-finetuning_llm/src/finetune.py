import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

# Setup FSDP plugin and accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Model and dataset configuration
base_model_id = "meta-llama/Meta-Llama-3-8B"
dataset_name = "scooterman/guanaco-llama3-1k"
new_model = "hpc_lt3_tutorial-llama3-8b-QLORA"

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load model and tokenizer with BnB quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_eos_token=True, add_bos_token=True
)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

try:
    # Ensure all parameters requiring gradients
    for param in model.parameters():
        param.requires_grad = True
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Output directory for results and checkpoints
output_dir = "./results"

# Training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=5,
    warmup_ratio=0.03,
    group_by_length=True,
    gradient_checkpointing=True,
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start training
trainer.train()

# Save the trained model
trainer.model.save_pretrained(new_model)
