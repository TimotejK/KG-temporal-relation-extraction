import os.path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from custom_datasets.combining_data import window_row_entity_bert
from custom_datasets.common import add_event_tokens
from custom_datasets.dataframe_dataset import DFDataset
from graph_building.graph_construction import link_to_umls
from graph_building.graph_construction import get_subgraph
from graph_building.llm import OpenChat
from graph_building.local_graph.build_local_patient_graph import create_graph

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
        print("Warning: missing llm response when only using cached responses")
        return None
    event1kg, response = OpenChat.get_kg_from_llm(event, "condition")
    llm_responses2[event] = (event1kg, response)
    torch.save(llm_responses2, "llm_responses2.pt")
    return event1kg

def generate_relation_graph_llm(row, **kwargs):
    event1 = row["event1_text"]
    event2 = row["event2_text"]
    relation = row["class"]

    graph1 = generate_llm_graph_for_event(event=event1, **kwargs)
    graph2 = generate_llm_graph_for_event(event=event2, **kwargs)
    if graph1 is None or graph2 is None:
        return None
    graph = combine_graphs(graph1=graph1, graph2=graph2, target=relation)
    return graph

def generate_primekg_graph_for_event(event, **kwargs):
    umls_id, mondo = link_to_umls(event)
    if umls_id is None:
        umls_id = event
    graph = get_subgraph(umls_id, event)
    return graph

def generate_relation_graph_primekg(row, **kwargs):
    event1 = row["event1_text"]
    event2 = row["event2_text"]
    relation = row["class"]

    graph1 = generate_primekg_graph_for_event(event=event1, **kwargs)
    graph2 = generate_primekg_graph_for_event(event=event2, **kwargs)
    if graph1 is None or graph2 is None:
        return None
    graph = combine_graphs(graph1=graph1, graph2=graph2, target=relation)
    return graph

def generate_local_graph_for_event(row, local_graph, configuration, **kwargs):
    graph = create_graph((row, local_graph, configuration))
    return graph

def combine_all_relation_graphs(llm_kg, local_kg, primekg_kg, row, **kwargs):
    global relation_types
    target = row["class"]

    if len(llm_kg.x.size()) < 2 or len(local_kg.x.size()) < 2 or len(primekg_kg.x.size()) < 2 or len(llm_kg.edge_attr.size()) < 2 or len(local_kg.edge_attr.size()) < 2 or len(primekg_kg.edge_attr.size()) < 2:
        print("Warning: empty nodes or edges")
        return None
    # pad to size
    entity_embedding_size = max(llm_kg.x.size()[1], local_kg.x.size()[1], primekg_kg.x.size()[1])
    # entity_embedding_size = 768
    edge_embedding_size = max(llm_kg.edge_attr.size()[1], local_kg.edge_attr.size()[1], primekg_kg.edge_attr.size()[1])
    # edge_embedding_size = 768

    # Padding
    llm_kg.x = F.pad(llm_kg.x, (0, entity_embedding_size - llm_kg.x.size()[1]), "constant", 0)
    llm_kg.edge_attr = F.pad(llm_kg.edge_attr, (0, edge_embedding_size - llm_kg.edge_attr.size()[1]), "constant", 0)
    local_kg.x = F.pad(local_kg.x, (0, entity_embedding_size - local_kg.x.size()[1]), "constant", 0)
    local_kg.edge_attr = F.pad(local_kg.edge_attr, (0, edge_embedding_size - local_kg.edge_attr.size()[1]), "constant", 0)
    primekg_kg.x = F.pad(primekg_kg.x, (0, entity_embedding_size - primekg_kg.x.size()[1]), "constant", 0)
    primekg_kg.edge_attr = F.pad(primekg_kg.edge_attr, (0, edge_embedding_size - primekg_kg.edge_attr.size()[1]), "constant", 0)

    # Combining
    x = torch.cat((llm_kg.x, local_kg.x, primekg_kg.x), 0)
    llm_num_nodes = llm_kg.x.size()[0]
    local_num_nodes = local_kg.x.size()[0]
    primekg_num_nodes = primekg_kg.x.size()[0]

    edge_index = torch.cat((llm_kg.edge_index,
                            local_kg.edge_index + llm_num_nodes,
                            primekg_kg.edge_index + llm_num_nodes + local_num_nodes), 1)
    edge_attr = torch.cat((llm_kg.edge_attr, local_kg.edge_attr, primekg_kg.edge_attr), 0)

    # reconnect all edges going to the event nodes to the events from the llm_kg
    edge_index[edge_index==local_kg.event1_index + llm_num_nodes] = llm_kg.event1_index
    edge_index[edge_index==primekg_kg.event1_index + llm_num_nodes + local_num_nodes] = llm_kg.event1_index
    edge_index[edge_index==local_kg.event2_index + llm_num_nodes] = llm_kg.event2_index
    edge_index[edge_index==primekg_kg.event2_index + llm_num_nodes + local_num_nodes] = llm_kg.event2_index

    return Data(x=x, y=torch.tensor([relation_types.index(target)]), edge_index=edge_index, edge_attr=edge_attr,
                event1_index=llm_kg.event1_index, event2_index=llm_kg.event2_index)

def generate_combination_graph(**kwargs):
    llm_kg = generate_relation_graph_llm(**kwargs)
    local_kg = generate_local_graph_for_event(**kwargs)
    primekg_kg = generate_relation_graph_primekg(**kwargs)
    if llm_kg is None or local_kg is None or primekg_kg is None:
        print("Warning: no graph provided for input!")
        return None
    return combine_all_relation_graphs(llm_kg=llm_kg, local_kg=local_kg, primekg_kg=primekg_kg, **kwargs)

def create_knowledge_graph_dataset(dataframe, graph_generation_function, **kwargs):
    def convert_row_to_graph(row, graph_generation_function, kwargs):
        classification_graph = graph_generation_function(row=row, **kwargs)
        if classification_graph is None:
            return None
        # use convert row
        row = window_row_entity_bert(row)
        if row is None:
            return None
        classification_graph["text"], classification_graph["event1_start"], classification_graph["event1_end"], classification_graph["event2_start"], classification_graph["event2_end"] \
            = add_event_tokens(row["text"], row["event1_start"], row["event1_end"], row["event2_start"], row["event2_end"])
        return classification_graph

    return DFDataset(dataframe, lambda row, args: convert_row_to_graph(row, graph_generation_function, args), kwargs)

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