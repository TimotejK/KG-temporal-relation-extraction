import torch
from transformers import BertTokenizer,BertModel

tokenizer = None
model = None
def sentence_embedding(text, type='bert'):
    global tokenizer, model
    if text is None or len(text.strip()) == 0:
        text = "empty"
    if type == 'bert':
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if model is None:
            model = BertModel.from_pretrained("bert-base-uncased")
        tokens = tokenizer(text, return_tensors='pt', max_length=512)
        output = model(**tokens)
        last_hidden_state, pooler_output = output[0], output[1]
        return pooler_output

if __name__ == '__main__':
    text = "This is a sample sentence."
    print(sentence_embedding(text))