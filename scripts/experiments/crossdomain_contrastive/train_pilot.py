"""Quick pilot: compare TCN baseline vs TCN+CrossDomain Contrastive on Real.

Uses a small subset of train cases to validate the approach before full training.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import CrossDomainContrastiveModel
from dataset import WindowDataset, LABEL_MAP
from losses import supervised_contrastive_loss

# Paths
CONTRACT = Path(r"D:\UAV-Data\feature_contract_v1.json")
NORM_STATS = Path(r"D:\UAV-Data\window_manifest_v1\norm_stats_train_v1.json")
OUTPUT_DIR = Path(r"D:\UAV-Data\experiments\crossdomain_contrastive_pilot")
PILOT_TRAIN = OUTPUT_DIR / "pilot_train.csv"
PILOT_VAL = OUTPUT_DIR / "pilot_val.csv"
PILOT_TEST = OUTPUT_DIR / "pilot_test.csv"
PILOT_REAL = OUTPUT_DIR / "pilot_external_test.csv"

# Pilot config
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
TEMPERATURE = 0.1
LAMBDA_CONTRAST = 0.5
SEED = 42


def make_balanced_sampler(dataset: WindowDataset) -> WeightedRandomSampler:
    """Create a sampler that balances classes and domains."""
    labels = dataset.records[dataset.label_field].map(LABEL_MAP).values
    class_counts = np.bincount(labels, minlength=3)
    weights_per_class = 1.0 / (class_counts + 1e-8)
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model on a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_domains = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["label"].numpy())
            all_domains.append(batch["domain"].numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_domains = np.concatenate(all_domains)

    # Per-class F1
    from sklearn.metrics import f1_score, classification_report
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Per-domain F1
    domain_f1 = {}
    for d_id, d_name in [(0, "SIL"), (1, "HIL"), (2, "Real")]:
        mask = all_domains == d_id
        if mask.sum() > 0:
            domain_f1[d_name] = float(f1_score(
                all_labels[mask], all_preds[mask], average="macro", zero_division=0
            ))

    return {
        "macro_f1": float(macro_f1),
        "normal_f1": float(per_class_f1[0]),
        "motor_f1": float(per_class_f1[1]),
        "sensor_f1": float(per_class_f1[2]),
        "domain_f1": domain_f1,
        "n_samples": len(all_labels),
    }


def train_model(
    use_contrastive: bool = True,
    tag: str = "full",
) -> dict:
    """Train and evaluate one model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {'TCN + SupCon CrossDomain' if use_contrastive else 'TCN baseline'}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Datasets (prefiltered CSVs for speed)
    train_ds = WindowDataset(
        PILOT_TRAIN, CONTRACT, NORM_STATS, prefiltered=True,
    )
    val_ds = WindowDataset(
        PILOT_VAL, CONTRACT, NORM_STATS, prefiltered=True,
    )
    real_ds = WindowDataset(
        PILOT_REAL, CONTRACT, NORM_STATS, prefiltered=True,
    )
    test_ds = WindowDataset(
        PILOT_TEST, CONTRACT, NORM_STATS, prefiltered=True,
    )

    print(f"Train: {len(train_ds)} windows")
    print(f"Val: {len(val_ds)} windows")
    print(f"Test (SIL+HIL): {len(test_ds)} windows")
    print(f"Real: {len(real_ds)} windows")

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = CrossDomainContrastiveModel(in_channels=26, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    best_val_f1 = 0.0
    best_state = None
    history = []

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_ce = 0.0
        epoch_con = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            labels = batch["label"].to(device)

            if use_contrastive:
                logits, z = model(x, return_proj=True)
                loss_ce = ce_loss_fn(logits, labels)
                loss_con = supervised_contrastive_loss(z, labels, TEMPERATURE)
                loss = loss_ce + LAMBDA_CONTRAST * loss_con
                epoch_con += loss_con.item()
            else:
                logits = model(x)
                loss_ce = ce_loss_fn(logits, labels)
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += loss_ce.item()
            n_batches += 1

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | CE={epoch_ce/n_batches:.4f} "
                f"Con={epoch_con/n_batches:.4f} | "
                f"Val F1={val_metrics['macro_f1']:.4f}"
            )

        history.append({
            "epoch": epoch,
            "ce_loss": epoch_ce / n_batches,
            "con_loss": epoch_con / n_batches if use_contrastive else 0,
            "val_macro_f1": val_metrics["macro_f1"],
        })

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Load best model and evaluate
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    real_metrics = evaluate(model, real_loader, device)

    print(f"\n  --- Results ({tag}) ---")
    print(f"  Test (SIL+HIL): macro-F1 = {test_metrics['macro_f1']:.4f}")
    print(f"    per-class: N={test_metrics['normal_f1']:.3f} M={test_metrics['motor_f1']:.3f} S={test_metrics['sensor_f1']:.3f}")
    if test_metrics["domain_f1"]:
        for d, f1 in test_metrics["domain_f1"].items():
            print(f"    {d}: {f1:.4f}")
    print(f"  Real (zero-shot): macro-F1 = {real_metrics['macro_f1']:.4f}")
    print(f"    per-class: N={real_metrics['normal_f1']:.3f} M={real_metrics['motor_f1']:.3f} S={real_metrics['sensor_f1']:.3f}")

    degradation = 0.0
    if test_metrics["macro_f1"] > 0:
        degradation = (test_metrics["macro_f1"] - real_metrics["macro_f1"]) / test_metrics["macro_f1"]
    print(f"  Cross-domain degradation: {degradation:.1%}")

    return {
        "tag": tag,
        "use_contrastive": use_contrastive,
        "n_params": n_params,
        "train_time_s": train_time,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "real_metrics": real_metrics,
        "degradation": degradation,
        "history": history,
    }


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train baseline (no contrastive)
    baseline_results = train_model(use_contrastive=False, tag="tcn_baseline")

    # Train with cross-domain contrastive
    contrastive_results = train_model(use_contrastive=True, tag="tcn_supcon")

    # Summary comparison
    print("\n" + "=" * 60)
    print("PILOT COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Test F1':<10} {'Real F1':<10} {'Degradation':<12}")
    print("-" * 52)
    for r in [baseline_results, contrastive_results]:
        print(
            f"{r['tag']:<20} "
            f"{r['test_metrics']['macro_f1']:<10.4f} "
            f"{r['real_metrics']['macro_f1']:<10.4f} "
            f"{r['degradation']:<12.1%}"
        )

    # Save results
    summary = {
        "config": {
            "max_windows_per_case": "prefiltered",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "temperature": TEMPERATURE,
            "lambda_contrast": LAMBDA_CONTRAST,
            "seed": SEED,
        },
        "baseline": {
            k: v for k, v in baseline_results.items() if k != "history"
        },
        "contrastive": {
            k: v for k, v in contrastive_results.items() if k != "history"
        },
    }
    (OUTPUT_DIR / "pilot_results.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nResults saved to {OUTPUT_DIR / 'pilot_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
