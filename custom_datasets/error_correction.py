import torch
import torch.nn.functional as F
def generate_edge_embedding(edge_type, edge_tensor, word_embedding_size=768):
    edge_type_index = ["date", "temporal_relation", "general_relation", "equivalent"].index(edge_type)
    edge_type_tensor = F.one_hot(torch.tensor(0), 4)

    if edge_type == "temporal_relation":
        return torch.cat((edge_type_tensor, edge_tensor[:3], torch.zeros(word_embedding_size)))
    elif edge_type == "date":
        # TODO use date2vec embeddings for time
        return torch.cat((edge_type_tensor, torch.zeros(3), edge_tensor))
    elif edge_type == "general_relation":
        return torch.cat((edge_type_tensor, torch.zeros(3), edge_tensor))
    elif edge_type == "equivalent":
        return torch.cat((edge_type_tensor, torch.zeros(3 + word_embedding_size)))
    raise Exception("Invalid edge type: " + edge_type)

def update_pregenerated_graph(graph):
    # recognise where relations came from
    edges_local_graph_start = 0
    edges_primekg_start = 0
    for edge_index in range(len(graph.edge_attr)):
        graph.edge_attr[edge_index]
        if graph.edge_attr[edge_index][3:].abs().sum() == 0:
            edges_primekg_start = edge_index
        elif edges_primekg_start == 0:
            edges_local_graph_start = edge_index
    edges_local_graph_start += 1
    edges_primekg_start += 1

    nodes_local_graph_start = 0
    nodes_primekg_start = 0
    for i in range(0, edges_local_graph_start):
        nodes_local_graph_start = max(nodes_local_graph_start, int(graph.edge_index[0][i]))
        nodes_local_graph_start = max(nodes_local_graph_start, int(graph.edge_index[1][i]))
    for i in range(edges_local_graph_start, edges_primekg_start):
        nodes_primekg_start = max(nodes_primekg_start, int(graph.edge_index[0][i]))
        nodes_primekg_start = max(nodes_primekg_start, int(graph.edge_index[1][i]))

    # add additional features
    edge_features = []
    for i in range(len(graph.edge_attr)):
        if i < edges_local_graph_start:
            # llm graph
            edge_type = 'general_relation'
        elif i >= edges_local_graph_start and i < edges_primekg_start:
            # local graph
            edge_type = 'temporal_relation'
        elif i >= edges_primekg_start:
            # primekg graph
            edge_type = 'general_relation'
        edge_features.append(generate_edge_embedding(edge_type, graph.edge_attr[i]))
    edge_attr = torch.cat([x.reshape(-1, 1) for x in edge_features], dim=1).T
    graph.edge_attr = edge_attr
    return graph

def fix_precomputed_dataset(dataset):
    for i in range(len(dataset.generated)):
        dataset.generated[i] = update_pregenerated_graph(dataset.generated[i])
    return dataset


if __name__ == '__main__':
    graph = torch.load("testni_graf.pt")
    print(update_pregenerated_graph(graph))