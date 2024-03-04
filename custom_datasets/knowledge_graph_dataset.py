import os.path

import torch
from torch_geometric.data import Data

from dataLoaders import combining_data
from dataset_loaders.dataframe_dataset import DFDataset
from graph_building.graph_construction import link_to_umls
from graph_building.graph_construction import get_subgraph
from graph_building.llm import OpenChat

relation_types = ["BEFORE", "AFTER", "OVERLAP"]
def combine_graphs(graph1, graph2, target):
    global relation_types
    x = torch.cat((graph1.x, graph2.x), 0)
    number_of_nodes_in_first_graph = graph1.x.size()[0]
    edge_index = torch.cat((graph1.edge_index, graph2.edge_index + number_of_nodes_in_first_graph), 1)
    edge_attr = torch.cat((graph1.edge_attr, graph2.edge_attr), 0)
    event1_index = graph1.term_index
    event2_index = graph2.term_index + number_of_nodes_in_first_graph
    # term_index = torch.cat((graph1.term_index, graph2.term_index + number_of_nodes_in_first_graph), 0)
    return Data(x=x, y=torch.tensor([relation_types.index(target)]), edge_index=edge_index, edge_attr=edge_attr, event1_index=event1_index, event2_index=event2_index)

llm_responses = {}
llm_responses2 = {}
if os.path.isfile("llm_responses.pt"):
    llm_responses = torch.load("llm_responses.pt")
if os.path.isfile("llm_responses2.pt"):
    llm_responses2 = torch.load("llm_responses2.pt")
def generate_llm_graph_for_event(event, **kwargs):
    global llm_responses
    if event in llm_responses:
        return llm_responses[event][0]
    if event in llm_responses2:
        return llm_responses2[event][0]
    if "cache_only" in kwargs and kwargs["cache_only"]:
        return None
    event1kg, response = OpenChat.get_kg_from_llm(event, "condition")
    llm_responses2[event] = (event1kg, response)
    torch.save(llm_responses2, "llm_responses2.pt")
    return event1kg

def generate_primekg_graph_for_event(event, **kwargs):

    umls_id, mondo = link_to_umls(event)
    if umls_id is None:
        umls_id = event
    graph = get_subgraph(umls_id, event)
    return graph

def generate_local_graph_for_event(event, **kwargs):

    pass

def create_knowledge_graph_dataset(dataframe, graph_generation_function, **kwargs):
    def convert_row_to_graph(row, graph_generation_function, kwargs):
        event1 = row['event1_text']
        event2 = row['event2_text']
        relation = row['class']
        event1kg = graph_generation_function(event=event1, row=row, **kwargs)
        event2kg = graph_generation_function(event=event2, row=row, **kwargs)
        if event1kg is None or event2kg is None:
            return None
        graph = combine_graphs(graph1=event1kg, graph2=event2kg, target=relation)
        graph["text"] = [row["text"]]
        graph["event1_start"] = [row["event1_start"]]
        graph["event1_end"] = [row["event1_end"]]
        graph["event2_start"] = [row["event2_start"]]
        graph["event2_end"] = [row["event2_end"]]
        return graph

    return DFDataset(dataframe, lambda row: convert_row_to_graph(row, graph_generation_function, kwargs))

def create_knowledge_graph_dataset_with_local_graphs(dataframe, graph_generation_function):
    def convert_row_to_graph(row, graph_generation_function):
        event1 = row['event1_text']
        event2 = row['event2_text']
        relation = row['class']
        event1kg = graph_generation_function(event1)
        event2kg = graph_generation_function(event2)
        if event1kg is None or event2kg is None:
            return None
        graph = combine_graphs(graph1=event1kg, graph2=event2kg, target=relation)
        graph["text"] = [row["text"]]
        graph["event1_start"] = [row["event1_start"]]
        graph["event1_end"] = [row["event1_end"]]
        graph["event2_start"] = [row["event2_start"]]
        graph["event2_end"] = [row["event2_end"]]
        return graph

    return DFDataset(dataframe, lambda row: convert_row_to_graph(row, graph_generation_function))

def get_llm_responses_only(df):
    number_of_rows = len(df)
    for i, row in df.iterrows():
        event1 = row['event1_text']
        event2 = row['event2_text']
        generate_llm_graph_for_event(event1)
        generate_llm_graph_for_event(event2)
        print(i, "/", number_of_rows)

if __name__ == '__main__':
    # i2b2df = combining_data.read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    # create_knowledge_graph_dataset(i2b2df)
    graph = generate_primekg_graph_for_event("hypertensive disorder")
    print(graph)