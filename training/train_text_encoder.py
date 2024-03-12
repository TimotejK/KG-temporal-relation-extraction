import evaluate
import numpy as np
import torch

from torch_geometric.data import DataLoader, Data
from transformers import Trainer, TrainingArguments

from custom_datasets.combining_data import read_i2b2
from custom_datasets.common import split_data
from custom_datasets.knowledge_graph_dataset import create_knowledge_graph_dataset, generate_llm_graph_for_event, \
    generate_relation_graph_llm
from models.knowledge_graph_encoder import GraphEncoder
from models.text_encoder import EntityBERTtextEncoder
from training.train_graph_encoder import prepare_dataset_combination_graph


def prepare_dataset_llm_only():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    dataset_train = create_knowledge_graph_dataset(df_train, generate_relation_graph_llm, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, generate_relation_graph_llm, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_val.pregenerate_and_filter()
    return dataset_train, dataset_val

def prepare_dataset_no_graph():
    def get_empty_graph(**kwargs):
        return Data(x=torch.empty((0,0)), y=torch.empty((0,0)), edge_index=torch.empty((0,0)), edge_attr=torch.empty((0,0)),
                    event1_index=0, event2_index=0)
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    dataset_train = create_knowledge_graph_dataset(df_train, get_empty_graph, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, get_empty_graph, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_val.pregenerate_and_filter()
    return dataset_train, dataset_val

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

def train():
    dataset_train, dataset_val = prepare_dataset_no_graph()

    model = EntityBERTtextEncoder(number_of_relations=3)
    # model = MultiModalPrediction(number_of_relations=3, combine_embeddings=True)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-2,
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

def hyper_parameter_search():
    dataset_train, dataset_val = prepare_dataset_no_graph()

    def model_init(trial):
        return EntityBERTtextEncoder(number_of_relations=3)

    def wandb_hp_space(trial):
        return {
            "name": "textsweep",
            "method": "random",
            "metric": {"name": "validation_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 0.001, "max": 0.5},
                "weight_decay": {"distribution": "uniform", "min": 0.001, "max": 0.5},
                "optimizer": {"values": ["sgd", "adam", "adamw"]}
            },
        }

    training_args = TrainingArguments(
        output_dir="./results_text",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        learning_rate=1e-2,
        weight_decay=1e-4,
        num_train_epochs=50,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_steps=5000,
        push_to_hub=False
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        model_init=model_init,
        data_collator=collate_function,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=100,
        # compute_objective=compute_objective,
    )
    return best_trial

if __name__ == '__main__':
    train()