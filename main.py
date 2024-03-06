import argparse

import torch
import sys

from custom_datasets.combining_data import read_i2b2
from custom_datasets.knowledge_graph_dataset import get_llm_responses_only
from training.train_graph_encoder import hyper_parameter_search, train

def prepare_llm_responses():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    get_llm_responses_only(df)

def run_graph_encoder_optimization():
    best_trial = hyper_parameter_search()
    torch.save(best_trial, "best_trial.pt")
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("--method", default="prepare_llm_responses")
    args = parser.parse_args()
    if args.method == "prepare_llm_responses":
        prepare_llm_responses()
    elif args.method == "train_graph":
        train()
