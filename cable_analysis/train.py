import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from .model import CableCNN
from .data import create_dataloaders


def train_model(csv_path: str, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32, output: str = "model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders(csv_path, batch_size=batch_size)
    model = CableCNN().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - val acc: {acc:.4f}")

    torch.save(model.state_dict(), output)
    print(f"Model saved to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Cable CNN model")
    parser.add_argument("csv", help="Path to training CSV file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="model.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.csv, args.epochs, args.lr, args.batch_size, args.output)
