import concurrent.futures
import threading

import torch
from torch.utils.data import Dataset

lock = threading.Lock()
class DFDataset(Dataset):
    def __init__(self, df=None, row_converter=None, args=None, save_path=None):
        self.df = df
        self.row_converter = row_converter
        self.generated = None
        self.args = args
        if save_path is not None:
            self.load(save_path)

    def __len__(self):
        if self.generated is not None:
            return len(self.generated)
        return len(self.df)

    def __getitem__(self, idx):
        if self.generated is not None:
            return self.generated[idx]
        return self.row_converter(self.df.iloc[idx], self.args)

    def save(self, path):
        if self.generated is None:
            raise Exception("Only dataframe with pregenerated data can be saved")
        torch.save((self.df, self.generated), path)

    def load(self, path):
        self.df, self.generated = torch.load(path)

    def graph_hash(self, graph):
        return hash((graph.text, graph.event1_start, graph.event1_end, graph.event2_start, graph.event2_end))
    def filter_out_repeated_entries(self):
        filtered_generated = []
        used_hashes = set()
        for graph in self.generated:
            hash = self.graph_hash(graph)
            if hash not in used_hashes:
                used_hashes.add(hash)
                filtered_generated.append(graph)

        self.generated = filtered_generated

    def pregenerate_and_filter(self):
        self.generated = [None for _ in self.df.iloc]
        environment = {"generated": self.generated, "df": self.df.iloc, "row_converter": self.row_converter, "args": self.args}
        def pretvori(ind, environment):
            environment["generated"][ind] = environment["row_converter"](environment["df"][ind], environment["args"])
            print("Graph", ind, "generated")


        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            print(f"\n ... executing workers ...\n")
            for i in range(len(self.df)):
                executor.submit(pretvori, i, environment)
        self.generated = [x for x in self.generated if x is not None]
        print("Generated", len(self.generated), "examples")