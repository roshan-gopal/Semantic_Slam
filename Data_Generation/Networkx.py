#The generation of data via the LLM remains the same. However, we must have a way of turning that inital dat
#gen into a graph in networkX


import networkx as nx
import random
from copy import deepcopy
from typing import Optional
from pprint import pprint
#import numpy as np
#from spine.mapping.graph_util import GraphHandler
#from spine.spine_util import UpdatePromptFormer

#We will mask the environment by removing and adding regions to the graph.
from graph_gen import Generate_Graph
import json
from typing import List, Optional

from openai import OpenAI

def create_graph(graph, json_object):
    print("creating graph")
    for region in json_object["graph"]["regions"]:
        graph.add_node(region["name"], info_dict = {"coords": region["coords"], "description": region["description"]})
        #add all the nodes
    
    for connection in json_object["graph"]["region_connections"]: 
        graph.add_edge(connection[0], connection[1])
        #add all the edges

    #How do we deal with the objects. Add them as nodes
    for object in json_object["graph"]["objects"]:
        graph.add_node(object["name"],  info_dict = {"coords": object["coords"], "description": object["description"]})

    for object_c in json_object["graph"]["object_connections"]:
        graph.add_edge(object_c[0], object_c[1])
    
    return graph

def get_full_adj_dict(graph):
    adj_dict = {}
    for node in graph.nodes():
        adj_dict[node] = list(graph.adj[node])
    return adj_dict

class Graph():
    def __init__(self, gpt_graph):
        graph = nx.Graph()
        self.graph = create_graph(graph, gpt_graph)
        self.full_adj_dict = get_full_adj_dict(self.graph)
        self.removed_nodes = []
        self.location = gpt_graph["graph"]["robot_location"]
        self.task = None

    def remove_nodes(self):
        nodes_to_remove  = []
        for node in self.graph.nodes():
            random_index = random.randint(0, 100)
            if random_index < 25: #25% chance of removal
                if self.location != node:
                    # Store both node name and attributes in the same format as creation
                    nodes_to_remove.append(node)
                    self.removed_nodes.append({"name": node, "info_dict": self.graph.nodes[node]["info_dict"]})

        self.graph.remove_nodes_from(nodes_to_remove)
            
        return self.graph


    def add_nodes(self):
        for node in self.removed_nodes:
            random_index = random.randint(0,100)
            if random_index < 95: #25% chance of addition

                self.graph.add_node(node["name"], info_dict = node["info_dict"])
                
                for item in self.full_adj_dict[node["name"]]:
                    if item in self.graph.nodes():
                        self.graph.add_edge(node["name"], item)

        return self.graph

def check_json_for_errors(json_string):
    if json_string.startswith("```json") and json_string.endswith("```"):
        json_string = json_string[7:-3]
    #Fix Json error
    if json_string.startswith("json") and json_string.endswith("json"):
        json_string = json_string[4:-4]
    return json_string

def get_observation(Graph):
    #get all the adjacency list for the location
    node = Graph.location 
    adj_list = Graph.full_adj_dict[node]

    to_gpt = {}
    for item in adj_list:
        if item in Graph.graph.nodes():
            print(Graph.graph.nodes[item])
            to_gpt["adj_locations/objects"][item] = Graph.graph.nodes[item]["info_dict"]
    
    to_gpt["current_location"] = node
    to_gpt["task"] = Graph.task

    return to_gpt
  


if __name__ == "__main__": 
    graph_gen = Generate_Graph()
    gpt_graph = graph_gen.generate(10, 10)
    print(gpt_graph)
    #We need to jsonify
    gpt_graph = check_json_for_errors(gpt_graph)
    gpt_graph = json.loads(gpt_graph)
    #Network X time

    Network_graph = Graph(gpt_graph)

    print(Network_graph.graph)
    print(Network_graph.full_adj_dict)

    new_graph = Network_graph.remove_nodes()
    print(Network_graph.graph)

    print(Network_graph.removed_nodes)

    added_nodes = Network_graph.add_nodes()
    print(Network_graph.graph)
    
    Network_graph.task = gpt_graph["tasks"][0]
    #give the robot a task in the actual implememtation
    observation = get_observation(Network_graph)

    print(observation)

