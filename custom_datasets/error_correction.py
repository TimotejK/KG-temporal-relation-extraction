import torch


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
        if i >= edges_local_graph_start and i < edges_primekg_start:
            edge_features.append(torch.cat((graph.edge_attr[i][:3], torch.zeros(768))))
        else:
            edge_features.append(torch.cat((torch.zeros(3), graph.edge_attr[i])))
    edge_attr = torch.cat([x.reshape(-1, 1) for x in edge_features], dim=1).T
    graph.edge_attr = edge_attr
    return graph



if __name__ == '__main__':
    graph = torch.load("testni_graf.pt")
    print(update_pregenerated_graph(graph))