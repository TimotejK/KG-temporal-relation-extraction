import torch
from transformers import BertTokenizer,BertModel
from transformers import Data2VecTextConfig, Data2VecTextModel

from custom_datasets.common import lock

tokenizer = None
model = None
def sentence_embedding(text, type='bert'):
    global tokenizer, model
    if text is None or len(text.strip()) == 0:
        text = "empty"
    if type == 'bert':
        with lock:
            if tokenizer is None:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            if model is None:
                model = BertModel.from_pretrained("bert-base-uncased")
        tokens = tokenizer(text, return_tensors='pt', max_length=512)
        output = model(**tokens)
        last_hidden_state, pooler_output = output[0], output[1]
        return pooler_output.detach()

def date_embedding(date):
    # Initializing a Data2VecText facebook/data2vec-text-base style configuration
    configuration = Data2VecTextConfig()
    # Initializing a model (with random weights) from the facebook/data2vec-text-base style configuration
    model = Data2VecTextModel(configuration)
    pass

if __name__ == '__main__':
    text = "This is a sample sentence."
    date_embedding(text)
    print(sentence_embedding(text))