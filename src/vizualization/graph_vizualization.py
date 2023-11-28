import numpy as np
import torch
from torch_geometric.data import Data

import torch_geometric
import networkx as nx

from src.data.graph_dataset import *
from src.data.load_spikegpt_embedding import *

def get_tokens(nodes_list, averaging=False):
    global emb, tokenizer
    result = []
    for el in nodes_list:
        embeddings = tokenize(el, tokenizer)
        result.append(embeddings.detach().tolist())
    return result

def replace_embedding(nodes_tokens):
    global emb
    res = []
    for n in nodes_tokens:
        res.append([])
        for i in n:
            print(i)
            res[-1].append(emb(i))
    return res

def create_graph_in_text(triplets, text):
    x = []
    edge_index = []
    edge_attr = []
    y = text

    for t in triplets:

        # Create nodes as unique words
        if t[0] not in x:
            x.append(t[0])
        if t[2] not in x:
            x.append(t[2])

        # Add edge
        ind1, ind2 = x.index(t[0]), x.index(t[2])
        edge_index.append([min(ind1, ind2), max(ind1, ind2)])

        # Add edge attribute
        edge_attr.append(t[1])

    edge_index = np.array(edge_index).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Do not store initial text for train sets
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=text)
    return data


def draw_graph(graph):
    data = create_graph_in_text(get_triplets(graph),graph['text'][0])
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw_shell(g, labels={i:p for i, p in enumerate(data.x)})