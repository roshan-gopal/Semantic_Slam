from copy import deepcopy
from typing import Optional
import random
from pprint import pprint
#import numpy as np
#from spine.mapping.graph_util import GraphHandler
#from spine.spine_util import UpdatePromptFormer

#We will mask the environment by removing and adding regions to the graph.
from graph_gen import Generate_Graph
import json
from typing import List, Optional

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#So far kind of works, but is not very robust. Fails when index of out range for add and remove regions, response often has trailing characters from GPT

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MaskEnvironment():
    def __init__(self, Generated_Graph , Original_Graph):
        self.Generated_Graph = Generated_Graph
        self.Generated_Graph_Regions = self.Generated_Graph["regions"]
        self.Generated_Graph_Regions_Length = len(self.Generated_Graph_Regions)
        self.Original_Graph = Original_Graph
        #Do the masking via the SPINE functions
    
    def remove_regions(self, current_region):
        random_index = random.randint(0, self.Generated_Graph_Regions_Length - 1)
        
        #The original graph should be stored
        self.Generated_Graph["regions"].pop(random_index)
        self.Generated_Graph_Regions.pop(random_index)
        
        return self.Generated_Graph

    
    def add_regions(self, current_region):
        #Fix this 
        random_index = random.randint(0, len(self.Original_Graph["regions"]) - 1)
#And finally came from here. Likely because the original graph regions is not a list.
        
        if self.Original_Graph["regions"][random_index] not in self.Generated_Graph["regions"]:
            self.Generated_Graph["regions"].append(self.Original_Graph["regions"][random_index])
        else:
            return False

def generate_plan(Temp_Graph, task, action_space):

    QUERY = f"""Based on the following {Temp_Graph} given above, generate a plan to complete the task. List actions you are going to take 
    at each step, and the region you are going to visit at each step. You can only visit regions that are connected to the current region and 
    can only do actions defined in this action space: {action_space}. Also list the region you will be at after each action. It must be a region that is connected to the current region. Provide your answer in the following JSON format:


    The task is: {task}

    Example:
    {{
    "plan": [
        {{"action": "action_1", "region": "region_1"}},
        {{"action": "action_2", "region": "region_2"}},
        ...
    ]
    }}

    Where the region is the region you will be at after the action. All reasoning should be assigned to a "reasoning" key. 
    There should be no random characters at the beginning or end of the response. Everything must be JSON.
    """

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": QUERY}
            ]
        )
    print("Response")
    print(response)
    plan_content = response.choices[0].message.content
    #We also need to parse the JSON here
    print("Plan content")
    print(plan_content)
    plan_string = check_json_for_errors(plan_content)
    print("Plan string")
    print(plan_string)
    plan_dict = json.loads(plan_string)
    return plan_dict

def get_observation(Generated_Graph, current_region):
    #The observation should be formatted as dictionary with keys of region: [name,description], object: [name,description]
    #current regions : [name,description], adjacent regions : [name,description]
    # Find current region info
    print("Current Region")
    print(current_region)
    current_region_info = next(
        (r for r in Generated_Graph["graph"]["regions"] if r["name"] == current_region),
        None
    )
    
    # Find objects connected to current region (from object_connections where region is the second element)
    objects_in_region = []
    for conn in Generated_Graph["graph"].get("object_connections", []):
        if len(conn) == 2 and conn[1] == current_region:
            obj_name = conn[0]
            obj_info = next(
                (o for o in Generated_Graph["graph"]["objects"] if o["name"] == obj_name),
                None
            )
            if obj_info:
                objects_in_region.append([obj_info["name"], obj_info.get("description", "")])
    
    # Find adjacent regions (from region_connections where current_region is either element)
    adjacent_regions = []
    for conn in Generated_Graph["graph"].get("region_connections", []):
        if len(conn) == 2:
            if conn[0] == current_region:
                adj_name = conn[1]
            elif conn[1] == current_region:
                adj_name = conn[0]
            else:
                continue
            
            adj_info = next(
                (r for r in Generated_Graph["graph"]["regions"] if r["name"] == adj_name),
                None
            )
            if adj_info:
                adjacent_regions.append([adj_info["name"], adj_info.get("description", "")])

    print(current_region_info["name"])

    observation = {
        "current_region": [current_region_info["name"], current_region_info.get("description", "")] if current_region_info else None,
        "objects": objects_in_region,
        "adjacent_regions": adjacent_regions
    }
    
    return observation
def ellicit_plan(Generated_Graph, num_iterations, task_number):
    # Generated_Graph should already be a dict
    #Generated graph remain the same throughout the iterations.

    #Action space 
    action_space= set([
            "explore_region",  # Explore area around a region
            "map_region",      # Map a specific region
            "inspect",         # Inspect an object/node
            "clarify",         # Ask for clarification
            "goto",            # Navigate to a location
            "answer",          # Provide final answer
            "extend_map",      # Extend map in a direction
            "replan",          # Replan strategy
    ])

    actions_and_observations = []
    current_region = Generated_Graph["graph"]["robot_location"]
    for i in range(num_iterations):

        print(Generated_Graph)
        task = Generated_Graph["tasks"][task_number]
        observation = get_observation(Generated_Graph, current_region)
        plan = generate_plan(observation, task, action_space)
        action = plan["plan"][0]["action"]
        Generated_Graph["graph"]["robot_location"] = plan["plan"][0]["region"]
        actions_and_observations.append({"action": action, "observation": observation})
        print("Actions and Observations")
        print(actions_and_observations)
        
        
    return actions_and_observations 

def check_json_for_errors(json_string):
    if json_string.startswith("```json") and json_string.endswith("```"):
        json_string = json_string[7:-3]
    #Fix Json error
    if json_string.startswith("json") and json_string.endswith("json"):
        json_string = json_string[4:-4]
    return json_string
  
if __name__ == "__main__":
    #Version 1 without masking!
    graph_gen = Generate_Graph()
    generated_graph_str = graph_gen.generate(10, 10)
    #This doesn't always give the same output. Sometimes the end and does not have ''' and the beginning doesn't have json??
    generated_graph_str = check_json_for_errors(generated_graph_str)
    #We need to parse JSON with JSON.loads
    
    generated_graph = json.loads(generated_graph_str)
    print("Generated Graph")
    print(generated_graph["graph"].keys())
    
   
   # Add tasks to the graph dict
   

    #Error happened here
    actions_and_observations = ellicit_plan(generated_graph, 10, 0)
    pprint(actions_and_observations)