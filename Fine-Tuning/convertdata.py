import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn.functional import relu
import random
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from math import sqrt
import json
from datasets import load_dataset

#we need to do embeddings before we make the dataset
def format_dataset(examples):
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

def convert_observation_to_string(example):
    # If observation is a dict, convert to JSON string
    if isinstance(example["prompt"], dict):
        example["prompt"] = json.dumps(example["prompt"])
    return example

class Dataset(Dataset):
    def __init__(self, action_observation_pairs):
        super().__init__()

        action_observation_pairs = action_observation_pairs.rename_column("observation", "prompt")
        action_observation_pairs = action_observation_pairs.rename_column("action", "completion")
        action_observation_pairs = action_observation_pairs.map(convert_observation_to_string)
        action_observation_pairs = action_observation_pairs.map(format_dataset)
        action_observation_pairs= action_observation_pairs.remove_columns(["prompt", "completion"])

        self.Data = action_observation_pairs

    
    def __len__(self):
        return len(self.Data)

# Adapted from trl.extras.dataset_formatting.instructions_formatting_function
# Converts dataset from prompt/completion format (not supported anymore)
# to the conversational format


if __name__ == "__main__":
    #test with one generated sample
    
    action_observation_pairs = load_dataset("json", data_files="Data/action_observation_pairs.json", split = "train")
    
    

    dataset = Dataset(action_observation_pairs)
    print(dataset.Data[0]['messages'])
    



