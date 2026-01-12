from transformers import AutoTokenizer, AutoModelForCausalLM
from model import LLAMAModel
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTConfig, SFTTrainer
from model import FinetuneModel
from peft import LoraConfig
from trl import SFTTrainer




from convertdata import Dataset
#Need to adjust is as currently it is not exporting as json
# Path relative to project root (run script from project root directory)
action_observation_pairs = load_dataset("json", data_files="Data/action_observation_pairs.json")



model = LLAMAModel()
tokenizer = AutoTokenizer.from_pretrained(model.repo_id)

dataset = Dataset(action_observation_pairs)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,    # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,  
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=16, 
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=True,

    ## GROUP 2: Dataset-related
    max_length=64, # renamed in v0.20
    # Dataset
    # packing a dataset means no padding is needed
    packing=True,
    packing_strategy='wrapped', # added to approximate original packing behavior

    ## GROUP 3: These are typical training parameters
    num_train_epochs=10,
    learning_rate=3e-4,
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='paged_adamw_8bit',       
    
    ## GROUP 4: Logging parameters
    logging_steps=10,
    logging_dir='./logs',
    output_dir='./OutputSLM',
    report_to='none',

    # ensures bf16 (the new default) is only used when it is actually available
    bf16=torch.cuda.is_bf16_supported(including_emulation=False)
)

FinetuneModel(dataset, tokenizer, model, sft_config, peft_params)







