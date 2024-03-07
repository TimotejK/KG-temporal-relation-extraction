import argparse

import torch
import sys

from custom_datasets.combining_data import read_i2b2
from custom_datasets.common import Configuration
from custom_datasets.knowledge_graph_dataset import get_llm_responses_only
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only
from training import train_text_encoder, train_graph_encoder
from training.train_graph_encoder import hyper_parameter_search, train

def prepare_llm_responses():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    get_llm_responses_only(df)

def run_graph_encoder_optimization():
    best_trial = hyper_parameter_search()
    torch.save(best_trial, "best_trial.pt")
    pass

def precompute_local_graphs():
    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.remove_target_relation = False
    configuration.use_realistic_graph = True
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    in_memory_kg = construct_graph_from_text_only(df, configuration, dataset_type="train")
    torch.save(in_memory_kg, "computed_kg.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("--method", default="prepare_llm_responses")
    args = parser.parse_args()
    if args.method == "prepare_llm_responses":
        prepare_llm_responses()
    elif args.method == "train_graph":
        train_graph_encoder.hyper_parameter_search()
    elif args.method == "train_text":
        train_text_encoder.hyper_parameter_search()
    elif args.method == "precompute_local_graphs":
        precompute_local_graphs()
