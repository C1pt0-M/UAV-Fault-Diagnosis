import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from uav_fault.models.cnn.bilstm_normalproto import BiLSTMNormalProtoClassifier
from uav_fault.models.cnn.focal_loss import FocalLoss


LABEL_NAMES = ["normal", "motor", "sensor"]
NORMAL_CLASS_ID = 0


class SequenceMetaDataset(Dataset):
    def __init__(self, cache_dir: str, norm_stats_path: str, augment: bool = False):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        self.augment = augment
        with open(norm_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).unsqueeze(1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], weights_only=True)
        x = (sample["x"] - self.mean) / self.std
        y = int(sample["y"])
        if self.augment:
            x = self._augment(x.clone())
        return x, y

    def _shift_no_wrap(self, x, shift):
        if shift == 0:
            return x
        out = torch.zeros_like(x)
        if shift > 0:
            out[:, shift:] = x[:, :-shift]
            out[:, :shift] = x[:, :1]
        else:
            k = -shift
            out[:, :-k] = x[:, k:]
            out[:, -k:] = x[:, -1:]
        return out

    def _augment(self, x):
        _, t = x.shape
        if np.random.random() < 0.35:
            shift = int(np.random.uniform(-0.02, 0.02) * t)
            x = self._shift_no_wrap(x, shift)
        if np.random.random() < 0.35:
            scale = 1.0 + np.random.uniform(-0.03, 0.03)
            x = x * scale
        if np.random.random() < 0.35:
            x = x + torch.randn_like(x) * 0.01
        return x


def build_model(model_name: str, in_channels: int, num_classes: int):
    if model_name == "bilstm_normalproto":
        return BiLSTMNormalProtoClassifier(
            in_channels=in_channels,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3,
        )
    raise ValueError(f"Unknown model: {model_name}")


def make_loader(cache_root: Path, split_names: list[str], batch_size: int, augment: bool, num_workers: int):
    norm_stats_path = cache_root / "norm_stats.json"
    datasets = []
    for split_name in split_names:
        split_dir = cache_root / split_name
        if split_dir.exists():
            datasets.append(
                SequenceMetaDataset(str(split_dir), str(norm_stats_path), augment=augment)
            )
    if not datasets:
        raise RuntimeError(f"未找到 split: {split_names}")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    names = [LABEL_NAMES[i] for i in labels]
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "labels": labels,
        "label_names": names,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=labels).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }


def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def split_recipe(mode: str):
    if mode == "fewshot":
        return {
            "train": ["train", "real_adapt_train"],
            "val": ["val", "real_adapt_val"],
            "test": ["test"],
            "external_test": ["real_holdout_test"],
        }
    raise ValueError(f"Unknown mode: {mode}")


def compute_normal_prototype_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    normal_center: torch.Tensor,
    normal_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal_mask = labels == NORMAL_CLASS_ID
    fault_mask = labels != NORMAL_CLASS_ID

    if normal_mask.any():
        pull = torch.mean((features[normal_mask] - normal_center.unsqueeze(0)) ** 2)
    else:
        pull = torch.tensor(0.0, device=features.device)

    if fault_mask.any():
        distances = torch.norm(
            features[fault_mask] - normal_center.unsqueeze(0), dim=1
        )
        push = torch.mean(F.relu(normal_margin - distances) ** 2)
    else:
        push = torch.tensor(0.0, device=features.device)

    return pull + push, pull, push


def print_metrics(metrics, split_name):
    print(f"\n{'=' * 52}")
    print(f"  {split_name}")
    print(f"{'=' * 52}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = split_recipe(args.mode)

    train_loader = make_loader(
        cache_root, recipe["train"], args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(
        cache_root, recipe["val"], args.batch_size, False, args.num_workers
    )
    test_loader = make_loader(
        cache_root, recipe["test"], args.batch_size, False, args.num_workers
    )
    ext_loader = make_loader(
        cache_root, recipe["external_test"], args.batch_size, False, args.num_workers
    )

    with open(cache_root / "norm_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    in_channels = len(stats["columns"])
    model = build_model(args.model, in_channels, len(LABEL_NAMES)).to(device)
    criterion = FocalLoss(gamma=args.gamma).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    trained_epochs = 0

    for epoch in range(args.epochs):
        lr = get_lr(epoch, args.warmup, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_proto_loss = 0.0
        total_pull = 0.0
        total_push = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            features = model.forward_features(x)
            logits = model.classify_features(features)
            cls_loss = criterion(logits, y)
            proto_loss, pull_loss, push_loss = compute_normal_prototype_loss(
                features,
                y,
                model.normal_center,
                args.normal_margin,
            )
            loss = cls_loss + args.lambda_normal * proto_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_proto_loss += proto_loss.item()
            total_pull += pull_loss.item()
            total_push += push_loss.item()
            steps += 1

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        trained_epochs = epoch + 1
        print(
            f"Epoch {trained_epochs:3d}/{args.epochs} | "
            f"loss={total_loss/max(1,steps):.4f} | "
            f"cls={total_cls_loss/max(1,steps):.4f} | "
            f"proto={total_proto_loss/max(1,steps):.4f} | "
            f"pull={total_pull/max(1,steps):.4f} | "
            f"push={total_push/max(1,steps):.4f} | "
            f"val_f1={val_f1:.4f} | lr={lr:.2e}"
        )

        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif trained_epochs >= args.min_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {trained_epochs} "
                    f"(best val_f1={best_val_f1:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    ext_metrics = evaluate(model, ext_loader, device)
    print_metrics(val_metrics, "Validation")
    print_metrics(test_metrics, "Test")
    print_metrics(ext_metrics, "External Test")

    torch.save(model.state_dict(), output_dir / "best.pt")
    for name, metrics in [
        ("val", val_metrics),
        ("test", test_metrics),
        ("external_test", ext_metrics),
    ]:
        (output_dir / f"metrics_{name}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    config = {
        "mode": args.mode,
        "model": args.model,
        "architecture": args.model,
        "task": "offline_multiclass",
        "num_classes": len(LABEL_NAMES),
        "label_names": LABEL_NAMES,
        "best_val_f1": best_val_f1,
        "epochs_trained": trained_epochs,
        "split_recipe": recipe,
        "in_channels": in_channels,
        "lambda_normal": args.lambda_normal,
        "normal_margin": args.normal_margin,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="transfer3 BiLSTM normal prototype training entrypoint"
    )
    parser.add_argument("--mode", choices=["fewshot"], required=True)
    parser.add_argument(
        "--model",
        choices=["bilstm_normalproto"],
        default="bilstm_normalproto",
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lambda-normal", dest="lambda_normal", type=float, default=0.1)
    parser.add_argument("--normal-margin", dest="normal_margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)


if __name__ == "__main__":
    main()
