import torch

from training.train_graph_encoder import hyper_parameter_search


def run_graph_encoder_optimization():
    best_trial = hyper_parameter_search()
    torch.save(best_trial, "best_trial.pt")
    pass

if __name__ == '__main__':
    run_graph_encoder_optimization()
