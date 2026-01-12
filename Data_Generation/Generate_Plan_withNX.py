from Networkx import *
from openai import OpenAI
import json
import os
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_plan(Graph, action_space, task):

    QUERY = f"""Based on the following {Graph} given above, generate a plan to complete the task. List actions you are going to take 
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
    There should be no random characters at the beginning or end of the response. When you write the region, just write the name 
    of the region instead of the full path. Example: 
    Write "path_1" if that is the location instead of "adj_locations/objects/path_1".  
    Everything must be JSON.
    """

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": QUERY}
            ]
        )
    
    plan_content = response.choices[0].message.content
    #We also need to parse the JSON here
    
    
    plan_string = check_json_for_errors(plan_content)
 
    
    plan_dict = json.loads(plan_string)
    return plan_dict

def get_observation(Graph):
    #get all the adjacency list for the location
    node = Graph.location 
    adj_list = Graph.full_adj_dict[node]

    to_gpt = {
        "adj_locations/objects" : {},
        "current_location": None,
        "task": None
    }
    for item in adj_list:
        if item in Graph.graph.nodes():
            print(Graph.graph.nodes[item])
            to_gpt["adj_locations/objects"][item] = Graph.graph.nodes[item]["info_dict"]
    
    to_gpt["current_location"] = node
    to_gpt["task"] = Graph.task

    return to_gpt

def elicit_plan(gpt_graph, num_iterations, task_number):  

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

    #parse the generation
    gpt_graph = check_json_for_errors(gpt_graph)
    gpt_graph = json.loads(gpt_graph)
    
    #Make the graph object
    Network_graph = Graph(gpt_graph)
    Network_graph.task = gpt_graph["tasks"][task_number]

    actions_and_observations = []

    for i in range(num_iterations):
        Network_graph.remove_nodes()
        Network_graph.add_nodes()
        observation = get_observation(Network_graph)
        plan = generate_plan(observation, action_space, Network_graph.task)
        action = plan["plan"][0]["action"]
        Network_graph.location = plan["plan"][0]["region"]
        actions_and_observations.append({"observation": observation, "action": action})
        print("Actions and Observations")
        print(actions_and_observations)

        
    return json.dumps(actions_and_observations, index = 2)

if __name__ == "__main__":
    graph_gen = Generate_Graph()
    gpt_graph = graph_gen.generate(10, 10)
    actions_and_observations = elicit_plan(gpt_graph, 10, 0)
    with open("Data/action_observation_pairs.json", "w") as f:
        f.write(actions_and_observations)
    pprint(actions_and_observations)