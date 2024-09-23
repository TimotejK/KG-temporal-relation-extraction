import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

modelcard = 'plenz/GLM-t5-large'

print('Load the model and tokenizer')
model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
tokenizer = AutoTokenizer.from_pretrained(modelcard, use_fast=True)

print('get dummy input (2 instances to show batching)')
graph_1 = [
    ('black poodle', 'is a', 'dog'),
    ('dog', 'is a', 'animal'),
    ('cat', 'is a', 'animal')
]
text_1 = 'The dog chased the cat.'

graph_2 = [
    ('dog', 'is a', 'animal'),
    ('dog', 'has', 'tail'),
    ('dog', 'has', 'fur'),
    ('fish', 'is a', 'animal'),
    ('fish', 'has', 'scales')
]
text_2 = None  # only graph for this instance

print('prepare model inputs')
how = 'global'  # can be 'global' or 'local', depending on whether the local or global GLM should be used. See paper for more details.
data_1 = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph_1, text=text_1, how=how)
data_2 = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph_2, text=text_2, how=how)
datas = [data_1, data_2]
model_inputs = model.data_processor.to_batch(data_instances=datas, tokenizer=tokenizer, max_seq_len=None, device='cpu')

print('compute token encodings')
outputs = model(**model_inputs)

# get token embeddings
print('Sequence of tokens (batch_size, max_seq_len, embedding_dim):',
      outputs.last_hidden_state.shape)  # embeddings of all graph and text tokens. Nodes in the graph (e.g., dog) appear only once in the sequence.
print('embedding of `black poodle` in the first instance. Shape is (seq_len, embedding_dim):',
      model.data_processor.get_embedding(sequence_embedding=outputs.last_hidden_state[0], indices=data_1.indices,
                                         concept='black poodle',
                                         embedding_aggregation='seq').shape)  # embedding_aggregation can be 'seq' or 'mean'. 'seq' returns the sequence of embeddings (e.g., all tokens of `black poodle`), 'mean' returns the mean of the embeddings.


class GraphLanguageModel(nn.Module):
    def __init__(self, number_of_relations=3, combine_embeddings=True):
        super(GraphLanguageModel, self).__init__()
        self.number_of_relations = number_of_relations
        modelcard = 'plenz/GLM-t5-large'
        self.model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(modelcard)
        self.mode = "global"  # global or local

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, labels):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data.to(device)
        text = data.text
        text_relations = data.text_relations

        datas = []
        for i in range(len(text)):
            data = model.data_processor.encode_graph(tokenizer=tokenizer, g=text_relations[i], text=text[i],
                                                     how=self.mode)
        model_inputs = model.data_processor.to_batch(data_instances=datas, tokenizer=tokenizer, max_seq_len=None,
                                                     device=device)
        outputs = model(**model_inputs)

        # bert_output = []
        # for i, layer in enumerate(x['last_hidden_state']):
        #     event1_emb = x['last_hidden_state'][i][
        #                  tokens[i].char_to_token(event1_start[i]):tokens[i].char_to_token(event1_end[i])]
        #     event1_emb = torch.mean(event1_emb, 0)
        #     event2_emb = x['last_hidden_state'][i][
        #                  tokens[i].char_to_token(event2_start[i]):tokens[i].char_to_token(event2_end[i])]
        #     event2_emb = torch.mean(event2_emb, 0)
        #     output = torch.cat((event1_emb, event2_emb))
        #     bert_output.append(output)
        # bert_output = torch.stack(bert_output)
        #
        # if return_embedding:
        #     return bert_output
        #
        # x = self.post_layers(bert_output)
        #     loss = self.criterion(x, labels)
        #     return {"loss": loss, "predictions": x}

        pass
