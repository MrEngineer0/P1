import argparse
import torch
import numpy as np

from .model import CableCNN

CATEGORY_MAP = {0: "geras", 1: "reikia papildomo bandymo", 2: "blogas"}


def load_model(model_path: str) -> CableCNN:
    model = CableCNN()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: CableCNN, sample: np.ndarray) -> str:
    with torch.no_grad():
        x = torch.from_numpy(sample).unsqueeze(0).float()
        out = model(x)
        idx = out.argmax(dim=1).item()
    return CATEGORY_MAP.get(idx, "unknown")


def parse_args():
    parser = argparse.ArgumentParser(description="Classify cable test sample")
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("sample", help="Path to numpy file with sample data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model)
    sample = np.load(args.sample)
    result = predict(model, sample)
    print(result)
