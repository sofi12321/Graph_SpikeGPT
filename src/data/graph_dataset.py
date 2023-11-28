import numpy as np
import torch
from torch_geometric.data import Data
import transform_text_pretrained_embedding
import torch
from torch_geometric.data import InMemoryDataset, download_url
import json
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

def create_graph(triplets, text, with_text=True):
    x = []
    edge_index = []
    y = text

    for t in triplets:

        # Create nodes as unique words
        if t[0] not in x:
            x.append(t[0])
        if t[2] not in x:
            x.append(t[2])

        # Add edge
        edge_index.append([x.index(t[0]), x.index(t[2])])

    x = get_tokens(x, averaging=True)
    k = 3
    # print(len(x))
    for i in range(len(x)):
        if len(x[i]) < k:
            x[i] += emb(torch.tensor( [1] * (k - len(x[i])))).detach().tolist()
        else:
            x[i] = x[i][:k]
        x[i] = sum(x[i], [])
    x = torch.tensor(x)

    edge_index = np.array(edge_index).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Do not store initial text for train sets
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data



class GraphDataset(InMemoryDataset):
    def __init__(self, root, file_path, transform=None, pre_transform=None, pre_filter=None):
        self.file_path = file_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.file_path]

    @property
    def processed_file_names(self):
        return [".".join(self.file_path.split(".")[:-1])+".pt"]

    def download(self):
        # Download to `self.raw_dir`.
        if ".pt" not in self.raw_file_names[0]:
            with open (self.raw_file_names[0], "r") as f:
                return json.load(f)
        else:
            return torch.load("processed/"+self.raw_file_names[0])

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if ".pt" in self.raw_file_names[0]:
            return
        data  = self.download()
        for datapoint in data:
            data_list.append(create_graph(get_triplets(datapoint), datapoint['text'][0]))
        # print("processing")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

if __name__ == "__main__":
    # Clone SpikeGPT to the src.third_party folder
    args = prepare_env()
    from scr.third_party.SpikeGPT.src.utils import TOKENIZER
    from scr.third_party.SpikeGPT.src.model_run import RWKV_RNN

    emb = load_embedding_weights("/content/SpikeGPT-OpenWebText-216M/", args)
    tokenizer = load_tokenizer()

    # data = create_graph(get_triplets(test_set[0]), test_set[0]['text'][0], with_text=True)
    train_dataset = GraphDataset("./", "/data/processed/train.pt")
    val_dataset = GraphDataset("./", "/data/processed/val.pt")
    test_dataset = GraphDataset("./", "/data/processed/test.pt")