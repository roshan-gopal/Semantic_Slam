import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

class LLAMAModel():
    def __init__(self):


        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32
        )
        self.repo_id = 'meta-llama/Llama-3.2-3B-Instruct'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id, device_map="cuda:0", quantization_config=self.bnb_config
        )

    def Lora(self, model):

        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
    # the rank of the adapter, the lower the fewer parameters you'll need to train
            r=8,                   
            lora_alpha=16, # multiplier, usually 2*r
            bias="none",           
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
    # Newer models, such as Phi-3 at time of writing, may require 
    # manually setting target modules
            target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
        )
        model = get_peft_model(model, config)

        return model


def FinetuneModel(Dataset, Tokenizer, Model, SFTConfig, PeftConfig): 
    trainer = SFTTrainer(
    model=Model.model, # the underlying Phi-3 model
    peft_config=PeftConfig,  # added to fix issue in TRL>=0.20
    processing_class=Tokenizer,
    args=SFTConfig,
    train_dataset=Dataset,
    )

    trainer.train()
    trainer.save_model()
    Tokenizer.save_pretrained(SFTConfig.output_dir) 

if __name__ == "__main__":
    model = LLAMAModel()
    print(model.model)
    print(model)
    model_lora = model.Lora(model.model)
    print(model_lora)
    




