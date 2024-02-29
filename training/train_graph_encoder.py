import numpy as np
import evaluate
from torch_geometric.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

from custom_datasets.common import split_data
from dataLoaders.combining_data import read_i2b2
from custom_datasets.knowledge_graph_dataset import generate_primekg_graph_for_event, create_knowledge_graph_dataset, \
    generate_llm_graph_for_event
from models.bimodal import MultiModalPrediction
from models.knowledge_graph_encoder import GraphEncoder

def prepare_dataset():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    dataset_train = create_knowledge_graph_dataset(df_train, lambda event: generate_llm_graph_for_event(event, cache_only=True))
    dataset_val = create_knowledge_graph_dataset(df_val, lambda event: generate_llm_graph_for_event(event, cache_only=True))
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
    dataset_train, dataset_val = prepare_dataset()

    model = GraphEncoder(node_size=768, edge_size=768, number_of_relations=3, dropout=0.2)
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
    dataset_train, dataset_val = prepare_dataset()

    def model_init(trial):
        return GraphEncoder(node_size=768, edge_size=768, number_of_relations=3, dropout=0.2)

    def wandb_hp_space(trial):
        return {
            "method": "random",
            "metric": {"name": "accuracy", "goal": "maximize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
                "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
                "optimizer": {"values": ["sgd", "adam", "adamw"]}
            },
        }

    training_args = TrainingArguments(
        output_dir="./results",
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
        data_collator=collate_function
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
    hyper_parameter_search()