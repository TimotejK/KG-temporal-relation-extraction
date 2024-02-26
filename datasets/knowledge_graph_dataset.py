import os.path

import torch
from torch_geometric.data import Data

from dataLoaders import combining_data
from graph_building.PrimeKG.PrimeKG_exploring import link_to_umls
from graph_building.PrimeKG.PrimeKG_exploring import get_subgraph
from graph_building.llm import OpenChat

relation_types = ["BEFORE", "AFTER", "OVERLAP"]
def combine_graphs(graph1, graph2, target):
    global relation_types
    x = torch.cat((graph1.x, graph2.x), 0).size()
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
def get_knowledge_graph_for_event(event):
    global llm_responses
    if event in llm_responses:
        return llm_responses[event][0]
    if event in llm_responses2:
        return llm_responses2[event][0]
    event1kg, response = OpenChat.get_kg_from_llm(event, "condition")
    llm_responses2[event] = (event1kg, response)
    torch.save(llm_responses2, "llm_responses2.pt")
    return event1kg

def generate_primekg_graph(event):
    umls_id, mondo = link_to_umls(event)
    if umls_id is None:
        umls_id = event
    graph = get_subgraph(umls_id, event)
    return graph

def create_knowledge_graph_dataset(dataframe):
    graphs = []
    for i, row in dataframe.iterrows():
        event1 = row['event1_text']
        event2 = row['event2_text']
        relation = row['class']
        event1kg = get_knowledge_graph_for_event(event1)
        event2kg = get_knowledge_graph_for_event(event2)
        graph = combine_graphs(graph1=event1kg, graph2=event2kg, target=relation)
        graph["text"] = [row["text"]]
        graph["event1_start"] = [row["event1_start"]]
        graph["event1_end"] = [row["event1_end"]]
        graph["event2_start"] = [row["event2_start"]]
        graph["event2_end"] = [row["event2_end"]]
        graphs.append(graph)
        pass
    torch.save(graphs, "i2b2_list_of_graphs.pt")

def get_llm_responses_only(df):
    number_of_rows = len(df)
    for i, row in df.iterrows():
        event1 = row['event1_text']
        event2 = row['event2_text']
        get_knowledge_graph_for_event(event1)
        get_knowledge_graph_for_event(event2)
        print(i, "/", number_of_rows)

if __name__ == '__main__':
    # i2b2df = combining_data.read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    # create_knowledge_graph_dataset(i2b2df)
    generate_primekg_graph("hypertensive disorder")