from collections import Counter

import nltk as nltk
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import nltk.data

from custom_datasets.combining_data import read_i2b2
from custom_datasets.common import Configuration, get_configuration_for_building_local_graph
from custom_datasets.error_correction import generate_edge_embedding
from custom_datasets.knowledge_graph_dataset import generate_relation_graph_llm, generate_relation_graph_primekg, \
    generate_local_graph_for_event, create_knowledge_graph_dataset, generate_combination_graph, \
    generate_fast_combination_graph
from graph_building import node_embeddings
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only

def extract_events(text):
    event_extraction_model = torch.load("event-model.pt", map_location=torch.device('cpu'))
    sentence_idxs = split_text_into_sentenes(text)
    events = []
    for sentence_idx in sentence_idxs:
        tokenized = event_extraction_model.tokenizer(text[sentence_idx[0]:sentence_idx[1]], return_tensors="pt")
        classification = event_extraction_model(tokenized)
        predictions = torch.argmax(classification["results"], axis=1)
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
                    events.append((sentence_idx[0] + start, sentence_idx[0] + end, text[sentence_idx[0] + start:sentence_idx[0] + end]))
                inside_event = False
        if inside_event:
            # end this event
            events.append((sentence_idx[0] + start, sentence_idx[0] + end, text[sentence_idx[0] + start:sentence_idx[0] + end]))
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
        sentence_index = get_sentence_index(sentences, start, end)
        event_sentence_indexes.append((sentence_index, event_idx))

    for sentence_idx in range(len(sentences)):
        events_from_this_sentence = list(map(lambda event_sent: events[event_sent[1]], filter(lambda event_sent: event_sent[0] == sentence_idx, event_sentence_indexes)))
        pairs += generate_all_event_pairs(events_from_this_sentence)
    return pairs


def get_sentence_index(sentences, start, end):
    sentence_index = 0
    for sentence_idx, (sent_start, sent_end) in enumerate(sentences):
        if sent_start <= start and sent_end >= end:
            sentence_index = sentence_idx
    return sentence_index


def construct_basic_dataframe(text, pairs, document_id):
    rows = []
    for pair in pairs:
        row = [
            text,
            "BEFORE",
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


def construct_dataset_with_graphs(text, dataframe, patient_id):
    configuration = get_configuration_for_building_local_graph()
    local_kg = construct_graph_from_text_only(dataframe, configuration, dataset_type="train")
    dataset = create_knowledge_graph_dataset(dataframe, generate_fast_combination_graph, configuration=configuration,
                                    local_graph=local_kg, cache_only=False, insert_time_nodes=True,
                                    graph_post_processing=add_stored_data_to_kg, patient_id=patient_id, relation_types=relation_types)
    dataset.pregenerate_and_filter()
    print(dataset)
    return dataset

relation_types = ["BEFORE", "AFTER", "OVERLAP"]
def predict_temporal_relations(dataset):
    human_readable_predictions = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.load("multimodal-model.pt", map_location=device)

    loader = DataLoader(dataset, batch_size=16)
    for batch in loader:
        batch.to(device)
        labels = batch.y
        result = model(data=batch, labels=labels)
        predictions = np.argmax(result["predictions"].cpu().detach().numpy(), axis=1)
        for i in range(len(batch["text"])):
            event1 = batch["text"][i][batch["event1_start"][i]:batch["event1_end"][i]]
            event2 = batch["text"][i][batch["event2_start"][i]:batch["event2_end"][i]]
            human_readable_predictions.append((event1, relation_types[predictions[i]], event2))
    return human_readable_predictions

patient_graphs = {}
global_graph = {}
def compute_event_pair_key(event1, event2):
    return (event1, event2)
def add_stored_data_to_kg(graph, patient_id, **kwargs):
    global patient_graphs, global_graph
    event1 = graph.text[graph.event1_start: graph.event1_end]
    event2 = graph.text[graph.event2_start: graph.event2_end]
    event_pair_key = compute_event_pair_key(event1, event2)
    edge_index_list = graph.edge_index.tolist()
    edge_attr_list = graph.edge_attr.tolist()
    if isinstance(graph.edge_type, list):
        edge_type_list = graph.edge_type
    else:
        edge_type_list = graph.edge_type.tolist()
    if patient_id in patient_graphs:
        patient_graph = patient_graphs[patient_id]
    else:
        patient_graph = {}
    if event_pair_key in global_graph:
        link_probabilities = [global_graph[event_pair_key]["BEFORE"],
                              global_graph[event_pair_key]["AFTER"],
                              global_graph[event_pair_key]["OVERLAP"]]
        link_probabilities = torch.tensor(link_probabilities)
        link_probabilities = link_probabilities / sum(link_probabilities)
        link_embedding = generate_edge_embedding("temporal_relation", 1, link_probabilities)
        edge_index_list[0].append(graph.event1_index)
        edge_index_list[1].append(graph.event2_index)
        edge_attr_list.append(link_embedding)
        edge_type_list.append(1)

    if event_pair_key in patient_graph:
        link_probabilities = [patient_graph[event_pair_key]["BEFORE"],
                              patient_graph[event_pair_key]["AFTER"],
                              patient_graph[event_pair_key]["OVERLAP"]]
        link_probabilities = torch.tensor(link_probabilities)
        link_probabilities /= sum(link_probabilities)
        link_embedding = generate_edge_embedding("temporal_relation", 1, link_probabilities)
        edge_index_list[0].append(graph.event1_index)
        edge_index_list[1].append(graph.event2_index)
        edge_attr_list.append(link_embedding)
        edge_type_list.append(1)
    graph.edge_index = torch.tensor(edge_index_list)
    graph.edge_attr = torch.tensor(edge_attr_list)
    graph.edge_type = torch.tensor(edge_type_list)
    return graph

def save_to_common_graph(predicted_relations, patient_id):
    global patient_graphs, global_graph
    if patient_id in patient_graphs:
        patient_graph = patient_graphs[patient_id]
    else:
        patient_graph = {}
    for relation in predicted_relations:
        events_key = compute_event_pair_key(relation[0], relation[2])
        if events_key not in patient_graph:
            patient_graph[events_key] = {"counter": Counter(), "event_mentions": (relation[0], relation[2])}
        patient_graph[events_key]["counter"].update([relation[1]])

        if events_key not in global_graph:
            global_graph[events_key] = {"counter": Counter(), "event_mentions": (relation[0], relation[2])}
        global_graph[events_key]["counter"].update([relation[1]])
    patient_graphs[patient_id] = patient_graph

def get_sentences_from_test_set_i2b2():
    all_documents = []
    all_sentences = []
    all_relations = []
    i2b2df = read_i2b2(full_text=True, use_test_files=True)
    documents = set(i2b2df["document_id"])
    for document in documents:
        relations = i2b2df[i2b2df["document_id"] == document].reset_index(drop=True)
        text = relations["text"][0]
        all_documents.append(text)
        sentences = split_text_into_sentenes(text)
        sentences_text = [text[s[0]: s[1]] for s in sentences]
        all_sentences.append(sentences_text)
        relations_per_sentence = [[] for _ in range(len(sentences))]
        for i, relation in relations.iterrows():
            sentence_index_1 = get_sentence_index(sentences, relation["event1_start"], relation["event1_end"])
            sentence_index_2 = get_sentence_index(sentences, relation["event2_start"], relation["event2_end"])
            if sentence_index_1 != sentence_index_2:
                # we ignore cross sentence relations
                continue
            relations_per_sentence[sentence_index_1].append((relation["event1_text"], relation["class"], relation["event2_text"]))
        all_relations.append(relations_per_sentence)
    return all_documents, all_sentences, all_relations


def most_simmilar_event_expanded(event_list :list[tuple[int, int, str]], event_target):
    min_difference = -1
    best_match = None
    for e in event_list:
        dif = -1
        event_text = e[2]
        if event_text in event_target:
            dif = len(event_target) - len(event_text)
        if event_target in event_text:
            dif = len(event_text) - len(event_target)
        if dif >= 0 and (min_difference < 0 or min_difference > dif):
            min_difference = dif
            best_match = e
    return best_match

def analyze_document(text, patient_id, event_pairs_of_interest=None):
    events = extract_events(text)
    print("Events:")
    print(events)
    if event_pairs_of_interest is None:
        event_pairs = generate_event_pairs(text, events)
    else:
        event_pairs = []
        for e1, e2 in event_pairs_of_interest:
            e1 = most_simmilar_event_expanded(events, e1)
            e2 = most_simmilar_event_expanded(events, e2)
            if e1 is not None and e2 is not None:
                event_pairs.append((e1, e2))
    print("Pairs (" + str(len(event_pairs)) + "):")
    print(event_pairs)
    dataframe = construct_basic_dataframe(text, event_pairs, 0)
    dataset = construct_dataset_with_graphs(text, dataframe, patient_id)
    relations = predict_temporal_relations(dataset)
    return relations

def run_pipeline():
    f = open("pipeline_predictions.txt", "a")
    documents, sentences, relations = get_sentences_from_test_set_i2b2()
    patient_id = 0
    for document, sentences, true_relations in zip(documents, sentences, relations):
        predicted_relations = analyze_document(document, patient_id)
        for i in range(len(sentences)):
            f.write(sentences[i] + "\n")
            f.write(str(true_relations[i]) + "\n")
            f.write(str(predicted_relations[i]) + "\n")
        f.flush()
        save_to_common_graph(predicted_relations, patient_id)
        patient_id += 1
    f.flush()
    f.close()

if __name__ == '__main__':
    # run_pipeline()
    analyze_document("He came to the hospital for a checkup on a mole that appeared two weeks ago.", 0)