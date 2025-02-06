import torch
from torch.utils.data import Dataset

class ECGDataset_1D(Dataset):
    def __init__(self, signal, label):
        self.signal = signal
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.signal[idx], self.label[idx]