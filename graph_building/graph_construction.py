import pandas as pd
import spacy
from tdc.resource import PrimeKG
# from scispacy.linking import EntityLinker
import torch
from torch_geometric.data import Data

from graph_building.node_embeddings import sentence_embedding

primeKG_data = None
umls_to_mondo = None


def get_PrimeKG():
    global primeKG_data, umls_to_mondo
    if primeKG_data is None:
        primeKG_data = PrimeKG(path='./PrimeKG/data')
        primeKG_data.to_nx()

    # nodes = pd.read_csv('./PrimeKG/github data/nodes.csv')
    # umls_codes = pd.read_csv('data/umls/umls.csv')

    if umls_to_mondo is None:
        umls_to_mondo = pd.read_csv('./PrimeKG/data/vocab/umls_mondo.csv')
    return primeKG_data, umls_to_mondo


nodes = None


def get_nodes():
    global nodes
    if nodes is None:
        nodes = pd.read_csv('./PrimeKG/github data/nodes.csv')
    return nodes


nlp = None


def link_to_umls(entity):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    primeKG_data, umls_to_mondo = get_PrimeKG()
    entities = nlp(entity)
    if len(entities.ents) > 0:
        for ent in entities.ents:
            cuid = str(ent)
            if cuid in list(umls_to_mondo['umls_id']):
                return cuid, int(umls_to_mondo.query('umls_id == "' + cuid + '"')['mondo_id'])
    return None


def linked_umls(cuid):
    _, umls_to_mondo = get_PrimeKG()
    if cuid in list(umls_to_mondo['umls_id']):
        return cuid, int(umls_to_mondo.query('umls_id == "' + cuid + '"')['mondo_id'])
    return None


disease_feature = None
def get_node_details(mondo):
    global disease_feature
    PrimeKG, _ = get_PrimeKG()
    if disease_feature is None:
        disease_feature = primeKG_data.get_features(feature_type='disease')
    nodes = get_nodes()
    features_disease = disease_feature.query('mondo_id == ' + str(mondo))
    definitions_and_descriptions = []
    for feature in features_disease.iloc:
        definitions_and_descriptions.append(feature['umls_description'])
        definitions_and_descriptions.append(feature['mondo_definition'])
    definitions_and_descriptions = list(dict.fromkeys(definitions_and_descriptions))
    names = []
    sources = []
    basic_data = nodes.query(
        'node_id == "' + str(mondo) + '"' + ' & (node_source == "MONDO_grouped" | node_source == "MONDO")')
    if len(basic_data) == 0:
        basic_data = nodes.query('node_id == "' + str(mondo) + '"')
    for feature in basic_data.iloc:
        names.append(feature['node_name'])
        sources.append(feature['node_source'])
    return {"definitions": definitions_and_descriptions, "name": names[0], "source": sources[0]}


def get_node_embedding(concept):
    return sentence_embedding(concept)


def get_link_embedding(relation):
    return sentence_embedding(relation)


def get_subgraph(entity):
    cuid, mondo = linked_umls(entity)
    details = get_node_details(mondo)
    links = primeKG_data.df.query('x_id == ' + str(mondo) + ' & x_source == "'+details['source']+'"')
    concepts = {mondo: details}
    concept_index = [mondo]
    relations = []
    for link in links.iloc:
        relations.append((link['relation'], link['x_id'], link['y_id']))
        target = link['y_id']
        if target not in concepts:
            concepts[target] = get_node_details(target)
            concept_index.append(target)

        pass

    display_graph(concepts, relations)

    # convert to torch geometric
    x = []
    edge_index = [[], []]
    edge_features = []
    for c in concept_index:
        x.append(get_node_embedding(c))
    for r in relations:
        edge_features.append(get_link_embedding(r[0]))
        edge_index[0].append(concept_index.index(r[1]))
        edge_index[1].append(concept_index.index(r[2]))

    data = Data(x=torch.Tensor(x), edge_index=torch.Tensor(edge_index), edge_attr=torch.Tensor(edge_features))
    return data


def display_graph(concepts, links):
    for link in links:
        c1 = concepts[link[1]]['name']
        c2 = concepts[link[2]]['name']
        link = link[0]
        print(c1, "--", link, "->", c2)

if __name__ == '__main__':
    print(get_subgraph("C0349644"))
    pass
