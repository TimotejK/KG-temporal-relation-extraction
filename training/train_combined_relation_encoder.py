import torch
from lightning import Trainer
from transformers import TrainingArguments

from models.bimodal import MultiModalPrediction


def train():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = MultiModalPrediction()
    text_model = torch.load("text-model.pt")
    graph_model = torch.load("graph_encoder.pt")
    model.text_model = text_model
    model.graph_model = graph_model

    model.to(device)
    # model = MultiModalPrediction(number_of_relations=3, combine_embeddings=True)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.001,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )
    # "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision" or "adafactor"
    training_args.set_optimizer(name="adafactor")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()
    torch.save(model, "multimodal-model.pt")
