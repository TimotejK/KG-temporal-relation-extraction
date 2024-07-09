import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class EventExtraction(nn.Module):
    def __init__(self, tokenizer):
        super(EventExtraction, self).__init__()
        self.bert = AutoModel.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
        self.tokenizer = tokenizer

    def forward(self, tokens, labels):
        embeddings = self.bert(tokens)
        pass