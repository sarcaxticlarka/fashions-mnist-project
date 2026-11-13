"""Training, validation, early stopping and evaluation routines for CNN models."""
import copy
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import CLASS_NAMES, SEED


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model: nn.Module, loader, device: torch.device) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader and return metrics dict, predictions, and true labels."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss, total_samples = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    avg_loss = total_loss / max(total_samples, 1)

    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro"))
    macro_prec = float(precision_score(targets, preds, average="macro"))
    per_class_rec = {c: float(r) for c, r in zip(CLASS_NAMES, recall_score(targets, preds, average=None))}

    m = {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "per_class_recall": per_class_rec,
    }
    return m, preds, targets


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    log_fn=print,
    seed: int = SEED,
    label_smoothing: float = 0.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train model with AdamW, Cosine Annealing, and Early Stopping on validation macro-F1."""
    set_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        tr_loss = train_loss / max(train_total, 1)
        tr_acc = train_correct / max(train_total, 1)

        val_metrics, _, _ = evaluate_model(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        log_fn(
            f"  Epoch {epoch:02d}/{epochs:02d}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_F1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_fn(f"  Early stopping triggered after {epoch} epochs (best val F1: {best_val_f1:.4f}).")
                break

    total_time = time.time() - t0_train
    model.load_state_dict(best_weights)
    return model, {
        "history": history,
        "best_val_f1": best_val_f1,
        "fit_seconds": round(total_time, 1),
    }


def measure_inference_latency(model: nn.Module, test_loader, device: torch.device, n_warmup=100, n_measure=2000):
    """Measure single-image inference latency on CPU using high-resolution monotonic clock."""
    cpu_device = torch.device("cpu")
    model_cpu = copy.deepcopy(model).to(cpu_device)
    model_cpu.eval()

    # Collect single-image tensors
    samples = []
    for x, _ in test_loader:
        for i in range(len(x)):
            samples.append(x[i : i + 1].to(cpu_device))
            if len(samples) >= n_warmup + n_measure:
                break
        if len(samples) >= n_warmup + n_measure:
            break

    with torch.no_grad():
        for s in samples[:n_warmup]:
            _ = model_cpu(s)

        t0 = time.perf_counter()
        for s in samples[n_warmup : n_warmup + n_measure]:
            _ = model_cpu(s)
        total_time = time.perf_counter() - t0

    latency_ms = (total_time / n_measure) * 1000.0
    return latency_ms

