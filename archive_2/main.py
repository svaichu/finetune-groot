import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    dataset_name: str
    split: str
    input_key: str
    target_key: str
    output_dir: str
    batch_size: int
    epochs: int
    lr: float
    hidden_sizes: list[int]
    val_ratio: float
    seed: int


class StateActionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, input_key: str, target_key: str) -> None:
        self.dataset = hf_dataset
        self.input_key = input_key
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[int(idx)]
        return (
            torch.tensor(item[self.input_key], dtype=torch.float32),
            torch.tensor(item[self.target_key], dtype=torch.float32),
        )


class PolicyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def build_dataloaders(cfg: TrainConfig):
    dataset = load_dataset(cfg.dataset_name, split=cfg.split)
    if cfg.val_ratio > 0:
        dataset = dataset.train_test_split(test_size=cfg.val_ratio, seed=cfg.seed)
        train_split = dataset["train"]
        val_split = dataset["test"]
    else:
        train_split = dataset
        val_split = None

    train_ds = StateActionDataset(train_split, cfg.input_key, cfg.target_key)
    val_ds = StateActionDataset(val_split, cfg.input_key, cfg.target_key) if val_split else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds else None
    return train_loader, val_loader, train_ds


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)
    return total_loss / max(total_count, 1)


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_ds = build_dataloaders(cfg)

    input_dim = train_ds[0][0].numel()
    output_dim = train_ds[0][1].numel()
    model = PolicyMLP(input_dim, output_dim, cfg.hidden_sizes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    os.makedirs(cfg.output_dir, exist_ok=True)
    config_path = os.path.join(cfg.output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(cfg.__dict__, handle, indent=2)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

        train_loss = total_loss / max(total_count, 1)
        log_line = f"epoch={epoch + 1} train_loss={train_loss:.6f}"
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            log_line += f" val_loss={val_loss:.6f}"
        print(log_line)

    model_path = os.path.join(cfg.output_dir, "policy_mlp.pt")
    torch.save(model.state_dict(), model_path)
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a policy on stack_without_ft_tact_v4.")
    parser.add_argument(
        "--dataset-name",
        default="LSY-lab/stack_without_ft_tact_v4",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-key", default="observation.state")
    parser.add_argument("--target-key", default="action")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        input_key=args.input_key,
        target_key=args.target_key,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train(config)
