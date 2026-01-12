import json
import os
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

QUERY = """
You are generating data for training an llm-based planner, like the SPINE paper from ravichandran et al.

Generate a scene graph for training in the following format

{
        "objects": [{"name": "object_1_name", "coords": [west_east_coordinate, south_north_coordinate]}, ...],
        "regions": [{"name": "region_1_name", "coords": [west_east_coordinate, south_north_coordinate]}, ...],
        "object_connections: [["object_name", "region_name"], ...],
        "region_connections": [["some_region_name", "other_region_name"], ...]
        "robot_location": "region_of_robot_location
}

For example,

{
"objects":
[
    {"name": "shed_1", "coords": [78, 9]},
    {"name": "gate_1", "coords": [52, -56]}
],
"regions": [
    {"name": "ground_1", "coords": [0, 0]},
    {"name": "road_1", "coords": [5.7, -8.3]},
    {"name": "road_2", "coords": [19.3, -6.5]},
    {"name": "road_3", "coords": [35.7, -12.1]},
    {"name": "road_4", "coords": [52.7, -20]},
    {"name": "road_5", "coords": [57.2, -31.6]},
    {"name": "bridge_1", "coords": [54.3, -46.7]},
    {"name": "road_6", "coords": [52.4, -56.5]},
    {"name": "driveway_1", "coords": [78.4, 9.1]}
],
"object_connections": [
    ["shed_1", "driveway_1"],
    ["gate_1", "road_6"]
],
"region_connections":[
    ["ground_1", "road_1"],
    ["road_1", "road_2"],
    ["road_2", "road_3"],
    ["road_3", "road_4"],
    ["road_4", "road_5"],
    ["road_5", "bridge_1"],
    ["bridge_1", "road_6"],
    ["road_6", "driveway_1"]
],
"robot_location": "ground_1"
}

Make sure all nodes referenced in the conntections are listed in the objects and regions list.
Provide your answer in the following JSON format:

{
reasoning: describe the type of scene you are creating,
graph: <JSON GRAPH>,
tasks: list of tasks that correspond to the graph.
}


Add a "description" attribute to each node that provides information.
These will be hidden from the robot

Task generation instructions
- DO NOT reference specific objects or nodes. Make the planner infer theese.
- Tasks should request specific information, not general exploration. Make the planner map or inspect certain entities. For example, start tasks with phrases such as "what", "I heard", "find out", "map", "inspect", "Can I", "is there", and likewise

"""

SCENE_PRIOR = """We are improving the SPINE planner proposed by ravichandran et al.
You need to generate data for training. Describe scenes you would train in, such as regions, objects, and general scene description

Describe ONE example environment, including scene, regions, and objects.
Such as `semi-urban office park with fields, roads, parking lots, buildings, people...` and more.

You will be randomly sampled, so be creative but realistic.

Your response should be a JSON with a "description" key, the value be the description.
"""

class Generate_Graph():
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def generate(self, n_regions, n_objects):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": QUERY +f"Generate a scene with {n_regions} regions and {n_objects} objects"}
            ]
        )
        return response.choices[0].message.content
    

        
if __name__ == "__main__":
    graph_gen = Generate_Graph()
    print(graph_gen.generate(10, 10))

