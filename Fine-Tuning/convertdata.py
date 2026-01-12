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

#we need to do embeddings before we make the dataset


class Dataset(Dataset):
    def __init__(self, actions_observation_pairs):
        super().__init__()
        
        for pair in actions_observation_pairs:
            pair["action"] = json.dumps(pair["action"])
            pair["observation"] = json.dumps(pair["observation"])

        self.Data = pd.DataFrame(actions_observation_pairs)
        self.Data = self.Data.to_numpy()
        
        

        self.targets = self.Data[:,0]
        self.features = self.Data[:,1]
    
    def __len__(self):
        return len(self.Data)


if __name__ == "__main__":
    #test with one generated sample
    actions_observation_pairs = [{"observation": {"adj_locations/objects": {"intersection_1": {"coords": [-5, 0], "description": "Intersection of different paths"}}, "current_location": "pathway_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "goto"}, {"observation": {"adj_locations/objects": {"pathway_1": {"coords": [0, 0], "description": "Main entrance pathway"}, "pathway_2": {"coords": [-5, 5], "description": "Leads to the playground"}, "pathway_3": {"coords": [5, 10], "description": "Connects to the pond"}, "pathway_5": {"coords": [-10, -5], "description": "Trail going to the garden"}}, "current_location": "intersection_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "goto"}, {"observation": {"adj_locations/objects": {"intersection_1": {"coords": [-5, 0], "description": "Intersection of different paths"}}, "current_location": "pathway_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "explore_region"}, {"observation": {"adj_locations/objects": {"pathway_1": {"coords": [0, 0], "description": "Main entrance pathway"}, "pathway_2": {"coords": [-5, 5], "description": "Leads to the playground"}, "pathway_3": {"coords": [5, 10], "description": "Connects to the pond"}, "pathway_5": {"coords": [-10, -5], "description": "Trail going to the garden"}}, "current_location": "intersection_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "map_region"}, {"observation": {"adj_locations/objects": {"pathway_1": {"coords": [0, 0], "description": "Main entrance pathway"}, "pathway_2": {"coords": [-5, 5], "description": "Leads to the playground"}, "pathway_3": {"coords": [5, 10], "description": "Connects to the pond"}, "pathway_5": {"coords": [-10, -5], "description": "Trail going to the garden"}}, "current_location": "intersection_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "explore_region"}, {"observation": {"adj_locations/objects": {"pathway_1": {"coords": [0, 0], "description": "Main entrance pathway"}, "pathway_2": {"coords": [-5, 5], "description": "Leads to the playground"}, "pathway_3": {"coords": [5, 10], "description": "Connects to the pond"}, "pathway_5": {"coords": [-10, -5], "description": "Trail going to the garden"}}, "current_location": "intersection_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "extend_map"}, {"observation": {"adj_locations/objects": {"pathway_1": {"coords": [0, 0], "description": "Main entrance pathway"}, "pathway_2": {"coords": [-5, 5], "description": "Leads to the playground"}, "pathway_3": {"coords": [5, 10], "description": "Connects to the pond"}, "pathway_5": {"coords": [-10, -5], "description": "Trail going to the garden"}}, "current_location": "intersection_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "goto"}, {"observation": {"adj_locations/objects": {"intersection_1": {"coords": [-5, 0], "description": "Intersection of different paths"}}, "current_location": "pathway_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "explore_region"}, {"observation": {"adj_locations/objects": {"intersection_1": {"coords": [-5, 0], "description": "Intersection of different paths"}}, "current_location": "pathway_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "goto"}, {"observation": {"adj_locations/objects": {"pathway_1": {"coords": [0, 0], "description": "Main entrance pathway"}, "pathway_2": {"coords": [-5, 5], "description": "Leads to the playground"}, "pathway_3": {"coords": [5, 10], "description": "Connects to the pond"}, "pathway_5": {"coords": [-10, -5], "description": "Trail going to the garden"}}, "current_location": "intersection_1", "task": "Inspect if there is a place to sit near the main entrance."}, "action": "goto"}]

    dataset = Dataset(actions_observation_pairs)
    print("First entry")
    print(dataset.Data[0])
    print("Targets")
    print(dataset.targets[0])
    print("Features")
    print(dataset.features[0])
    print("all data for json")
    print(dataset.Data)



