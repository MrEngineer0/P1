from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
torch = None

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover - torch might not be available
    pass


class CableDataset(torch.utils.data.Dataset):
    """Dataset reading PD, FDS, PDC, RMV values from a CSV file."""

    def __init__(self, csv_path: str, seq_length: int = 100):
        data = pd.read_csv(csv_path)
        features = data[['PD', 'FDS', 'PDC', 'RMV']].values
        labels = data['label'].values
        self.labels = labels.astype(np.int64)

        # assuming features are provided sequentially for each sample
        # reshape to [num_samples, 4, seq_length]
        self.samples = features.reshape(len(features), 4, seq_length)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.samples[idx], self.labels[idx]


def create_dataloaders(csv_path: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    dataset = CableDataset(csv_path)
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
