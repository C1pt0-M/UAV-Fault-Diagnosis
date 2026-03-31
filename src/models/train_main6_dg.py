import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.autograd import Function
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from cnn.cnn_lstm import CNNLSTMClassifier
from cnn.focal_loss import FocalLoss
from cnn.textcnn import TextCNN1d
from cnn.transformer import PositionalEncoding


LABEL_NAMES = ["normal", "motor", "propeller", "sensor", "wind", "load"]
DOMAIN_NAMES = ["SIL", "HIL", "Real"]


def get_warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


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
        domain_y = int(sample.get("domain_y", -1))
        meta = {
            "filename": sample.get("filename", self.files[idx].name),
            "platform": sample.get("platform", ""),
            "sample_id": sample.get("sample_id", ""),
            "domain_y": domain_y,
        }
        if self.augment:
            x = self._augment(x.clone())
        return x, y, domain_y, meta

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
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        if np.random.random() < 0.15:
            mask_len = int(np.random.uniform(0.02, 0.05) * t)
            start = np.random.randint(0, max(1, t - mask_len + 1))
            x[:, start:start + mask_len] = 0.0
        return x


class FeatureTransformer(nn.Module):
    def __init__(self, in_channels=31, d_model=128, nhead=4, num_layers=4, patch_size=8, dropout=0.2):
        super().__init__()
        self.embed_conv = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.embed_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        max_patches = 2048 // patch_size + 1
        self.pos_enc = PositionalEncoding(d_model, max_len=max_patches, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_norm = nn.LayerNorm(d_model)

    def forward_features(self, x):
        n = x.size(0)
        x = self.embed_conv(x)
        x = x.transpose(1, 2)
        x = self.embed_norm(x)
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        return self.feature_norm(cls_out)


class DGTransformerClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, num_domains=3, d_model=128, nhead=4, num_layers=4, patch_size=8, dropout=0.2):
        super().__init__()
        self.backbone = FeatureTransformer(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_domains),
        )

    def forward(self, x, alpha=0.0, return_domain=False):
        feat = self.backbone.forward_features(x)
        cls_logits = self.cls_head(feat)
        if not return_domain:
            return cls_logits
        rev_feat = ReverseLayerF.apply(feat, alpha)
        domain_logits = self.domain_head(rev_feat)
        return cls_logits, domain_logits


def build_model(model_name: str, in_channels: int, num_classes: int):
    if model_name == "transformer":
        return DGTransformerClassifier(in_channels=in_channels, num_classes=num_classes)
    if model_name == "textcnn":
        return TextCNN1d(in_channels=in_channels, num_classes=num_classes, num_filters=128, kernel_sizes=(3, 5, 7, 11, 21), dropout=0.3)
    if model_name == "cnn_lstm":
        return CNNLSTMClassifier(
            in_channels=in_channels,
            cnn_channels=(64, 128, 128),
            lstm_hidden=128,
            lstm_layers=2,
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
            datasets.append(SequenceMetaDataset(str(split_dir), str(norm_stats_path), augment=augment))
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


def evaluate(model, loader, device, mode: str):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            logits = model(x) if mode in {"baseline", "fewshot"} else model(x, alpha=0.0, return_domain=False)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    present_labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    target_names = [LABEL_NAMES[i] for i in present_labels]
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "labels": present_labels,
        "label_names": target_names,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=present_labels).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def print_metrics(metrics, split_name):
    print(f"\n{'=' * 52}")
    print(f"  {split_name}")
    print(f"{'=' * 52}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")


def build_split_recipe(mode: str):
    if mode == "baseline":
        return {
            "train": ["train"],
            "val": ["val"],
            "test": ["test"],
            "external_test": ["external_test"],
        }
    if mode == "fewshot":
        return {
            "train": ["train", "real_adapt_train"],
            "val": ["val", "real_adapt_val"],
            "test": ["test"],
            "external_test": ["real_holdout_test"],
        }
    if mode == "dg":
        return {
            "train": ["train"],
            "val": ["val"],
            "test": ["test"],
            "external_test": ["external_test"],
        }
    if mode == "dg_fewshot":
        return {
            "train": ["train", "real_adapt_train"],
            "val": ["val", "real_adapt_val"],
            "test": ["test"],
            "external_test": ["real_holdout_test"],
        }
    raise ValueError(f"Unknown mode: {mode}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_recipe = build_split_recipe(args.mode)
    train_loader = make_loader(cache_root, split_recipe["train"], args.batch_size, augment=True, num_workers=args.num_workers)
    val_loader = make_loader(cache_root, split_recipe["val"], args.batch_size, augment=False, num_workers=args.num_workers)
    test_loader = make_loader(cache_root, split_recipe["test"], args.batch_size, augment=False, num_workers=args.num_workers)
    ext_loader = make_loader(cache_root, split_recipe["external_test"], args.batch_size, augment=False, num_workers=args.num_workers)

    with open(cache_root / "norm_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    in_channels = len(stats["columns"])

    model = build_model(args.model, in_channels, len(LABEL_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"设备: {device}")
    print(f"模式: {args.mode}")
    print(f"模型: {args.model}")
    print(f"参数量: {n_params:,}")

    criterion = FocalLoss(gamma=args.gamma).to(device)
    domain_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    train_losses = []
    val_f1s = []

    for epoch in range(args.epochs):
        lr = get_warmup_cosine_lr(epoch, args.warmup, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        alpha = min(1.0, (epoch + 1) / max(1, args.domain_warmup_epochs))

        for x, y, domain_y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            domain_y = domain_y.to(device)

            optimizer.zero_grad()
            if args.mode in {"dg", "dg_fewshot"} and args.model == "transformer":
                cls_logits, domain_logits = model(x, alpha=alpha, return_domain=True)
                cls_loss = criterion(cls_logits, y)
                valid_mask = domain_y >= 0
                dom_loss = domain_criterion(domain_logits[valid_mask], domain_y[valid_mask]) if valid_mask.any() else torch.tensor(0.0, device=device)
                loss = cls_loss + args.domain_lambda * dom_loss
            else:
                cls_logits = model(x)
                cls_loss = criterion(cls_logits, y)
                dom_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device, args.mode)
        val_f1 = val_metrics["macro_f1"]
        train_losses.append(avg_loss)
        val_f1s.append(val_f1)

        ep = epoch + 1
        print(f"Epoch {ep:3d}/{args.epochs} | loss={avg_loss:.4f} | val_f1={val_f1:.4f} | lr={lr:.2e}")

        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif ep >= args.min_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {ep} (best val_f1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device, args.mode)
    test_metrics = evaluate(model, test_loader, device, args.mode)
    ext_metrics = evaluate(model, ext_loader, device, args.mode)

    print_metrics(val_metrics, "Validation")
    print_metrics(test_metrics, "Test")
    print_metrics(ext_metrics, "External Test")

    torch.save(model.state_dict(), output_dir / "best.pt")
    (output_dir / "metrics_val.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics_test.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics_external_test.json").write_text(json.dumps(ext_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "mode": args.mode,
        "model": args.model,
        "in_channels": in_channels,
        "num_classes": len(LABEL_NAMES),
        "label_names": LABEL_NAMES,
        "epochs_trained": len(train_losses),
        "best_val_f1": best_val_f1,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_params": n_params,
        "domain_lambda": args.domain_lambda,
        "domain_warmup_epochs": args.domain_warmup_epochs,
        "split_recipe": split_recipe,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n模型和结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="main_6class + Real generalization 训练入口")
    parser.add_argument("--mode", choices=["baseline", "fewshot", "dg", "dg_fewshot"], required=True)
    parser.add_argument("--model", choices=["transformer", "textcnn", "cnn_lstm"], default="transformer")
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
    parser.add_argument("--domain-lambda", type=float, default=0.2)
    parser.add_argument("--domain-warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)


if __name__ == "__main__":
    main()
