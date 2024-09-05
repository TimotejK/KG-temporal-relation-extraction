import os.path

import numpy as np
import evaluate
import torch
from torch_geometric.data import DataLoader
from transformers import Trainer, TrainingArguments

from custom_datasets.common import split_data, get_configuration_for_building_local_graph
from custom_datasets.combining_data import read_i2b2, window_row_entity_bert
from custom_datasets.dataframe_dataset import DFDataset
from custom_datasets.error_correction import fix_precomputed_dataset, generate_edge_embedding
from custom_datasets.knowledge_graph_dataset import create_knowledge_graph_dataset, \
    generate_llm_graph_for_event, generate_combination_graph, generate_relation_graph_llm, \
    generate_fast_combination_graph
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only
from models.bimodal import MultiModalPrediction
from models.knowledge_graph_encoder import GraphEncoder
from models.text_encoder import EntityBERTtextEncoder


def prepare_dataset_combination_graph(balanced=True):
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_test = read_i2b2(full_text=True, use_test_files=True, include_rows_without_absolute=True)
    df_train, df_val, df_val2 = split_data(df, oversample=False, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    configuration = get_configuration_for_building_local_graph()

    patient_graphs_train = construct_graph_from_text_only(df_train, configuration, dataset_type="train")
    patient_graphs_val = construct_graph_from_text_only(df_val, configuration, dataset_type="val")
    patient_graphs_test = construct_graph_from_text_only(df_test, configuration, dataset_type="test")

    dataset_train = create_knowledge_graph_dataset(df_train, generate_fast_combination_graph, configuration=configuration,
                                                   local_graph=patient_graphs_train, cache_only=False, insert_time_nodes=True)
    dataset_val = create_knowledge_graph_dataset(df_val, generate_fast_combination_graph, configuration=configuration,
                                                   local_graph=patient_graphs_val, cache_only=False, insert_time_nodes=True)
    dataset_test = create_knowledge_graph_dataset(df_test, generate_fast_combination_graph, configuration=configuration,
                                                   local_graph=patient_graphs_test, cache_only=False, insert_time_nodes=True)

    dataset_train.pregenerate_and_filter()
    dataset_train.save("pregenerated/dataset_train_new.pt")
    dataset_val.pregenerate_and_filter()
    dataset_val.save("pregenerated/dataset_val_new.pt")
    dataset_test.pregenerate_and_filter()
    dataset_test.save("pregenerated/dataset_test_new.pt")

    if balanced:
        dataset_train.oversample_pregenerated()
        dataset_val.oversample_pregenerated()

    return dataset_train, dataset_val, dataset_test

def load_stored_dataset_combination_graph(balanced=True):
    dataset_train = DFDataset()
    dataset_train.load("pregenerated/dataset_train_new.pt")
    dataset_val = DFDataset()
    dataset_val.load("pregenerated/dataset_val_new.pt")
    dataset_test = DFDataset()
    dataset_test.load("pregenerated/dataset_test_new.pt")

    if balanced:
        dataset_train.oversample_pregenerated()
        dataset_val.oversample_pregenerated()

    return dataset_train, dataset_val, dataset_test

def collate_function(examples):
    loader = DataLoader(examples, batch_size=len(examples))
    batch = next(iter(loader))
    return {"data": batch, "labels": batch.y}

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_objective(eval_pred):
    return eval_pred["eval_accuracy"]

def window_text(graph):
    graph = window_row_entity_bert(graph, normalize_event_order=False)
    if graph is None:
        return None
    return graph


device = "cuda:0" if torch.cuda.is_available() else "cpu"
def train_graph(dataset_train, dataset_val, dataset_test):
    model = GraphEncoder(node_size=768, edge_size=768 + 7, number_of_relations=3, dropout=0.2)

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results-graph",
        learning_rate=2e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()

    torch.save(model, "evaluation_results/graph_encoder.pt")

    results = trainer.evaluate(eval_dataset=dataset_val)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("Results on eval - graph:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_test)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("Results on test - graph:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    return model

def train_bimodal(dataset_train, dataset_val, dataset_test, graph_model):
    model = MultiModalPrediction(number_of_relations=3, combine_embeddings=True)
    model.graph_model = graph_model
    model.to(device)

    dataset_train.generated = list(filter(lambda x: x is not None, map(window_text, dataset_train.generated)))
    dataset_val.generated = list(filter(lambda x: x is not None, map(window_text, dataset_val.generated)))
    dataset_test.generated = list(filter(lambda x: x is not None, map(window_text, dataset_test.generated)))

    training_args = TrainingArguments(
        output_dir="./results-bimodal",
        learning_rate=0.001,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.0001,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()

    torch.save(model, "evaluation_results/bimodal-model.pt")

    results = trainer.evaluate(eval_dataset=dataset_val)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("Results on eval - bimodal" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_test)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("Results on test - bimodal:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    return model

def train_text(dataset_train, dataset_val, dataset_test):
    model = EntityBERTtextEncoder(number_of_relations=3, pooling_strategy='both_events')
    model.to(device)

    dataset_train.generated = list(filter(lambda x: x is not None, map(window_text, dataset_train.generated)))
    dataset_val.generated = list(filter(lambda x: x is not None, map(window_text, dataset_val.generated)))
    dataset_test.generated = list(filter(lambda x: x is not None, map(window_text, dataset_test.generated)))

    training_args = TrainingArguments(
        output_dir="./results-text",
        learning_rate=0.001,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.0001,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()

    torch.save(model, "evaluation_results/text-model.pt")

    results = trainer.evaluate(eval_dataset=dataset_val)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("Results on eval - text" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_test)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("Results on test - text:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    return model

def train():
    # dataset_train, dataset_val, dataset_test = prepare_dataset_combination_graph()
    dataset_train, dataset_val, dataset_test = load_stored_dataset_combination_graph()
    # graph_model = train_graph(dataset_train, dataset_val, dataset_test)
    graph_model = torch.load("evaluation_results/graph_encoder.pt")
    bimodal_model = train_bimodal(dataset_train, dataset_val, dataset_test, graph_model)
    text_model = train_text(dataset_train, dataset_val, dataset_test)

if __name__ == '__main__':
    train()