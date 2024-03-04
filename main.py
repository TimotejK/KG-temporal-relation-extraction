import torch

from graph_building.local_graph.build_local_patient_graph import precompute_local_knowledge_graph
from training.train_graph_encoder import hyper_parameter_search


def run_graph_encoder_optimization():
    best_trial = hyper_parameter_search()
    torch.save(best_trial, "best_trial.pt")
    pass

if __name__ == '__main__':
    precompute_local_knowledge_graph()
