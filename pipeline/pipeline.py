import nltk as nltk
import pandas as pd
import torch
from transformers import AutoTokenizer
import numpy as np
import nltk.data

from custom_datasets.common import Configuration
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only

def extract_events(text):
    event_extraction_model = torch.load("event-model.pt", map_location=torch.device('cpu'))
    tokenized = event_extraction_model.tokenizer(text, return_tensors="pt")
    classification = event_extraction_model(tokenized)
    predictions = torch.argmax(classification["results"], axis=1)
    events = []
    start = 0
    end = 0
    inside_event = False
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            if tokenized.token_to_chars(i) is None:
                continue
            if not inside_event:
                start = tokenized.token_to_chars(i).start
            end = tokenized.token_to_chars(i).end
            inside_event = True
        else:
            if inside_event:
                # end this event
                events.append((start, end, text[start:end]))
            inside_event = False
    if inside_event:
        # end this event
        events.append((start, end, text[start:end]))
    return events

def split_text_into_sentenes(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = list(tokenizer.span_tokenize(text))
    return sentences

def generate_all_event_pairs(events):
    pairs = []
    for i in range(len(events) - 1):
        for j in range(i+1, len(events)):
            pairs.append((events[i], events[j]))
    return pairs

def generate_event_pairs(text, events):
    pairs = []
    # get event pars that appear in the same sentence
    sentences = split_text_into_sentenes(text)
    event_sentence_indexes = []
    for event_idx, event in enumerate(events):
        start, end, event_text = event
        sentence_index = 0
        for sentence_idx, (sent_start, sent_end) in enumerate(sentences):
            if sent_start<=start and sent_end >= end:
                sentence_index = sentence_idx
        event_sentence_indexes.append((sentence_index, event_idx))

    for sentence_idx in range(len(sentences)):
        events_from_this_sentence = list(map(lambda event_sent: events[event_sent[1]], filter(lambda event_sent: event_sent[0] == sentence_idx, event_sentence_indexes)))
        pairs += generate_all_event_pairs(events_from_this_sentence)
    return pairs

def construct_basic_dataframe(text, pairs, document_id):
    rows = []
    for pair in pairs:
        row = [
            text,
            None,
            pair[0][0],
            pair[0][1],
            None,
            pair[1][0],
            pair[1][1],
            None,
            pair[0][2],
            pair[1][2],
            document_id,
            "eval",
            None
        ]
        rows.append(row)
    return pd.DataFrame(rows, columns=['text', 'class', 'event1_start', 'event1_end', 'event1_type', 'event2_start', 'event2_end', 'event2_type', 'event1_text', 'event2_text', 'document_id', 'source', 'additional_document_info'])
    pass


def construct_graphs(text, dataframe):
    # TODO
    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.remove_target_relation = False
    configuration.use_realistic_graph = True
    graph = construct_graph_from_text_only(dataframe, configuration, dataset_type="train")
    print(graph)

def predict_temporal_relation(pair):
    # TODO
    pass

def analyze_document(text):
    events = extract_events(text)
    print(events)
    event_pairs = generate_event_pairs(text, events)
    print(event_pairs)
    dataframe = construct_basic_dataframe(text, event_pairs, 0)
    construct_graphs(text, dataframe)
    pass

if __name__ == '__main__':
    analyze_document("He came to the hospital for a checkup on a mole that appeared two weeks ago.")