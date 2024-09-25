import os.path
from collections import Counter
from datetime import datetime

import numpy as np
import evaluate
import torch
from torch_geometric.data import DataLoader
from transformers import Trainer, TrainingArguments
import cryptpandas as crp

from custom_datasets.common import split_data, get_configuration_for_building_local_graph
from custom_datasets.combining_data import read_i2b2, window_row_entity_bert
from custom_datasets.dataframe_dataset import DFDataset
from custom_datasets.error_correction import fix_precomputed_dataset, generate_edge_embedding
from custom_datasets.knowledge_graph_dataset import create_knowledge_graph_dataset, \
    generate_llm_graph_for_event, generate_combination_graph, generate_relation_graph_llm, \
    generate_fast_combination_graph
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only
from models.baselines.GPTmodel import GPTTemporalRelationExtraction
from models.bimodal import MultiModalPrediction
from models.knowledge_graph_encoder import GraphEncoder
from models.text_encoder import EntityBERTtextEncoder


def prepare_dataset_combination_graph(balanced=True, dataset="i2b2"):
    if dataset == "i2b2":
        relation_types = ["BEFORE", "AFTER", "OVERLAP"]
        df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
        df_test = read_i2b2(full_text=True, use_test_files=True, include_rows_without_absolute=True)
    elif dataset == "thyme":
        relation_types = ['BEFORE', 'AFTER', 'OVERLAP', 'BEGINS-ON', 'CONTAINED-BY', 'CONTAINS', 'ENDS-ON', 'CONTINUES', 'TERMINATES', 'INITIATES', 'REINITIATES']
        df = crp.read_encrypted(path='thyme.crypt', password=os.environ['TP'])
        df_test = crp.read_encrypted(path='thyme_test.crypt', password=os.environ['TP'])
    df_train, df_val, df_val2 = split_data(df, oversample=False, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    configuration = get_configuration_for_building_local_graph()

    patient_graphs_train = construct_graph_from_text_only(df_train, configuration, dataset_type="train")
    patient_graphs_val = construct_graph_from_text_only(df_val, configuration, dataset_type="val")
    patient_graphs_test = construct_graph_from_text_only(df_test, configuration, dataset_type="test")

    dataset_val = create_knowledge_graph_dataset(df_val, generate_fast_combination_graph, configuration=configuration,
                                                   local_graph=patient_graphs_val, cache_only=False, insert_time_nodes=True, relation_types=relation_types)
    dataset_train = create_knowledge_graph_dataset(df_train, generate_fast_combination_graph, configuration=configuration,
                                                   local_graph=patient_graphs_train, cache_only=False, insert_time_nodes=True, relation_types=relation_types)
    dataset_test = create_knowledge_graph_dataset(df_test, generate_fast_combination_graph, configuration=configuration,
                                                   local_graph=patient_graphs_test, cache_only=False, insert_time_nodes=True, relation_types=relation_types)

    dataset_train.pregenerate_and_filter()
    dataset_train.save("pregenerated/"+dataset+"_dataset_train_rawkg.pt")
    dataset_val.pregenerate_and_filter()
    dataset_val.save("pregenerated/"+dataset+"_dataset_val_rawkg.pt")
    dataset_test.pregenerate_and_filter()
    dataset_test.save("pregenerated/"+dataset+"_dataset_test_rawkg.pt")

    if balanced:
        dataset_train.oversample_pregenerated()
        dataset_val.oversample_pregenerated()

    return dataset_train, dataset_val, dataset_test

def load_stored_dataset_combination_graph(balanced=True, dataset="i2b2"):
    dataset_train = DFDataset()
    dataset_train.load("pregenerated/"+dataset+"_dataset_train_rawkg.pt")
    dataset_val = DFDataset()
    dataset_val.load("pregenerated/"+dataset+"_dataset_val_rawkg.pt")
    dataset_test = DFDataset()
    dataset_test.load("pregenerated/"+dataset+"_dataset_test_rawkg.pt")

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

def train_universal(model, dataset_steps, training_args_steps, model_description):
    model.to(device)
    for i in range(len(dataset_steps)):
        dataset_train, dataset_val, dataset_test = dataset_steps[i]
        dataset_train.generated = list(filter(lambda x: x is not None, map(window_text, dataset_train.generated)))
        dataset_val.generated = list(filter(lambda x: x is not None, map(window_text, dataset_val.generated)))
        dataset_test.generated = list(filter(lambda x: x is not None, map(window_text, dataset_test.generated)))

        training_args = training_args_steps[i]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=collate_function,
            compute_metrics=compute_metrics
        )
        trainer.train()

        results = trainer.evaluate(eval_dataset=dataset_val)
        with open("evaluation_results/results.txt", "a") as myfile:
            myfile.write(model_description + " - midpoint results on eval:" + "\n")
            myfile.write(str(results) + "\n")
            myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_val)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write(model_description + " - End results - validation:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_test)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write(model_description + " - End results - test:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()
    return model

def train_graph(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, number_of_relations, test_name):
    model = GraphEncoder(node_size=768, edge_size=768 + 7, number_of_relations=number_of_relations, dropout=0.2)

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

    model = train_universal(model,
                            [(dataset_train, dataset_val, dataset_test_ub),
                             (dataset_train_ub, dataset_val_ub, dataset_test_ub)],
                            [training_args, training_args], "Graph")

    torch.save(model, "evaluation_results/graph_encoder-"+test_name+".pt")

    return model

def test_gpt_model(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub,number_of_relations, test_name):
    model = GPTTemporalRelationExtraction()

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
    results = trainer.evaluate(eval_dataset=dataset_val)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("ChatGPT" + " - End results - val:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_test_ub)
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("ChatGPT" + " - End results - test:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()
    return model


def test_baseline_model(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, test_name):
    dataset = dataset_val
    if dataset is not None:
        classes = [int(a.y) for a in dataset.generated]
        number_of_most_common_apperances = Counter(classes).most_common(1)[0][1]
        accuracy = number_of_most_common_apperances / len(dataset.generated)
        with open("evaluation_results/results.txt", "a") as myfile:
            myfile.write("Baseline" + " - balanced - val:" + "\n")
            myfile.write(str(accuracy) + "\n")
            myfile.flush()

    dataset = dataset_val_ub
    if dataset is not None:
        classes = [int(a.y) for a in dataset.generated]
        number_of_most_common_apperances = Counter(classes).most_common(1)[0][1]
        accuracy = number_of_most_common_apperances / len(dataset.generated)
        with open("evaluation_results/results.txt", "a") as myfile:
            myfile.write("Baseline" + " - unbalanced - val:" + "\n")
            myfile.write(str(accuracy) + "\n")
            myfile.flush()

    dataset = dataset_test_ub
    if dataset is not None:
        classes = [int(a.y) for a in dataset.generated]
        number_of_most_common_apperances = Counter(classes).most_common(1)[0][1]
        accuracy = number_of_most_common_apperances / len(dataset.generated)
        with open("evaluation_results/results.txt", "a") as myfile:
            myfile.write("Baseline" + " - unbalanced - test:" + "\n")
            myfile.write(str(accuracy) + "\n")
            myfile.flush()

def train_glm(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, graph_model, number_of_relations, test_name):
    model = MultiModalPrediction(number_of_relations=number_of_relations, combine_embeddings=True)
    model.graph_model = graph_model

    training_args = TrainingArguments(
        output_dir="./results-glm",
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

    model = train_universal(model,
                            [(dataset_train, dataset_val, dataset_test_ub), (dataset_train_ub, dataset_val_ub, dataset_test_ub)],
                            [training_args, training_args], "Bimodal")

    torch.save(model, "evaluation_results/bimodal-model-"+test_name+".pt")
    return model

def train_bimodal(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, graph_model, number_of_relations, test_name):
    model = MultiModalPrediction(number_of_relations=number_of_relations, combine_embeddings=True)
    model.graph_model = graph_model

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

    model = train_universal(model,
                            [(dataset_train, dataset_val, dataset_test_ub), (dataset_train_ub, dataset_val_ub, dataset_test_ub)],
                            [training_args, training_args], "Bimodal")

    torch.save(model, "evaluation_results/bimodal-model-"+test_name+".pt")
    return model

def train_text(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, number_of_relations, test_name):
    model = EntityBERTtextEncoder(number_of_relations=number_of_relations, pooling_strategy='both_events')

    training_args = TrainingArguments(
        output_dir="./results-text",
        learning_rate=0.0001,
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
    model = train_universal(model,
                            [(dataset_train, dataset_val, dataset_test_ub),
                             (dataset_train_ub, dataset_val_ub, dataset_test_ub)],
                            [training_args, training_args], "Text")

    torch.save(model, "evaluation_results/text-model-"+test_name+".pt")

    return model

def train():
    with open("evaluation_results/results.txt", "a") as myfile:
        myfile.write("\nTest " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        myfile.flush()

    dataset_train, dataset_val, dataset_test = prepare_dataset_combination_graph(balanced=True, dataset="thyme")
    # dataset_train, dataset_val, _ = load_stored_dataset_combination_graph(balanced=True, dataset="thyme")
    dataset_train_ub, dataset_val_ub, dataset_test_ub = load_stored_dataset_combination_graph(balanced=False, dataset="thyme")
    # number_of_relations = 9
    number_of_relations = 11 # imamo relacije 0, 2, 3, 5, 6, 7, 8, 9, 10
    test_name = "thyme"

    # graph_model = test_gpt_model(None, dataset_val, None, None, dataset_test_ub)
    # graph_model = torch.load("evaluation_results/graph_encoder.pt")
    # graph_model = train_graph(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, number_of_relations=number_of_relations, test_name=test_name)
    # bimodal_model = train_bimodal(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, graph_model, number_of_relations=number_of_relations, test_name=test_name)
    # text_model = train_text(dataset_train, dataset_val, dataset_train_ub, dataset_val_ub, dataset_test_ub, number_of_relations=number_of_relations, test_name=test_name)
    test_baseline_model(None, dataset_val, None, None, dataset_test_ub, test_name=test_name)
    test_gpt_model(None, dataset_val, None, None, dataset_test_ub, number_of_relations=number_of_relations, test_name=test_name)

if __name__ == '__main__':
    # train()
    prepare_dataset_combination_graph(balanced=True, dataset="i2b2")
    # with open("evaluation_results/results.txt", "a") as myfile:
    #     myfile.write("\nTest " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    #     myfile.flush()
    dataset_train, dataset_val, dataset_test_ub = load_stored_dataset_combination_graph(balanced=True)
    test_gpt_model(None, dataset_val, None, None, dataset_test_ub)