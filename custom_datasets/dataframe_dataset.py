from torch.utils.data import Dataset

class DFDataset(Dataset):
    def __init__(self, df, row_converter):
        self.df = df
        self.row_converter = row_converter
        self.generated = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.generated is not None:
            return self.generated[idx]
        return self.row_converter(self.df.iloc[idx])

    def pregenerate_and_filter(self):
        self.generated = []
        for row in self.df.iloc:
            converted = self.row_converter(row)
            if converted is not None:
                self.generated.append(converted)