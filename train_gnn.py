from __future__ import annotations

import hashlib
import json
import logging
import sys
import time

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import typer

from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from qqe.GNN.physics_aware_NN import GNN, QuantumCircuitGraphDataset
from qqe.utils import configure_logger

# Family registry for validation
FAMILY_REGISTRY = {
    "haar": True,
    "clifford": True,
    "quansistor": True,
    "random": True,
}


def _amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger(__name__)
device_type = _amp_device_type()
logger.info(f"AMP device type: {device_type}")


def collect_pt_paths(dataset_dir: str, family: str | None = None) -> list[str]:
    d = Path(dataset_dir)
    # support either dataset_dir/*.pt or dataset_dir/samples/*.pt
    if family is not None:
        paths = sorted((d / "encoding_data_pennylane" / family).glob("*.pt"))
    else:
        paths = sorted((d / "encoding_data_pennylane").glob("*.pt"))
    if not paths:
        paths = sorted(d.glob("*.pt"))
    return [str(p) for p in paths]


def _cache_root_for_paths(paths: list[str], suffix: str = "") -> str:
    canonical = "|".join(sorted(Path(p).name for p in paths))

    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]

    tag = f"_{suffix}" if suffix else ""

    cache_dir = Path("qqe") / "cache" / f"pyg_cache_{digest}{tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return str(cache_dir)


class PaddedGraphDatasetWrapper:
    """Wrapper that pads all graphs to a consistent node feature dimension."""

    def __init__(self, dataset, target_dim: int | None = None):
        self.dataset = dataset
        # Compute max dimension across entire dataset if not provided
        if target_dim is None:
            self.target_dim = self._compute_max_dim()
        else:
            self.target_dim = target_dim

    def _compute_max_dim(self) -> int:
        """Find the max node feature dimension across all samples."""
        max_dim = 0
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            if hasattr(data, "x") and data.x.dim() > 1:
                max_dim = max(max_dim, data.x.shape[1])
        return max_dim

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        if hasattr(data, "x") and data.x.shape[1] < self.target_dim:
            pad_size = self.target_dim - data.x.shape[1]
            # Create a new copy to avoid mutating cache
            data = data.clone()
            data.x = torch.nn.functional.pad(data.x, (0, pad_size), mode="constant", value=0)
        return data

    def __len__(self) -> int:
        return len(self.dataset)


def _safe_y(batch) -> torch.Tensor:
    """Return y as float tensor shaped [num_graphs]."""
    if not hasattr(batch, "y") or batch.y is None:
        raise RuntimeError("Batch is missing labels 'y'. Make sure your dataset sets data.y.")
    y = batch.y
    if y.dim() == 0:
        y = y.view(1)
    if y.dim() == 2 and y.size(1) == 1:
        y = y.view(-1)
    return y.float()


def get_node_feature_dim_from_sample(pt_paths: list[str]) -> int:
    obj = torch.load(pt_paths[0], map_location="cpu")
    return int(obj["x"].shape[1])


def get_global_feature_dim_from_sample(pt_paths: list[str]) -> int:
    obj = torch.load(pt_paths[0], map_location="cpu")
    return int(obj["global_features"].numel())


def build_train_test_loaders(
    pt_paths: list[str],
    train_split: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pt_paths, suffix=suffix)

    dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    if len(dataset) < 2:
        raise RuntimeError("Dataset too small to split. Check PT paths.")

    # Wrap dataset to handle padding
    padded_dataset = PaddedGraphDatasetWrapper(dataset)

    train_len = max(1, int(len(padded_dataset) * train_split))
    test_len = max(1, len(padded_dataset) - train_len)
    while train_len + test_len > len(padded_dataset):
        train_len -= 1

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(
        padded_dataset,
        [train_len, test_len],
        generator=generator,
    )

    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_mem,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        ),
    )


def build_full_loader(
    pt_paths: list[str],
    batch_size: int = 64,
    global_feature_variant: str = "binned",
    node_feature_backend_variant: str | None = None,
) -> DataLoader:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pt_paths, suffix=suffix)

    dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check PT paths and formats.")

    # Wrap dataset to handle padding
    padded_dataset = PaddedGraphDatasetWrapper(dataset)

    pin_mem = torch.cuda.is_available()

    return DataLoader(
        padded_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_mem,
    )


def build_train_val_test_loaders_two_stage(
    pt_paths: list[str],
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: str | None = None,
):
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pt_paths, suffix=suffix)

    dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    if len(dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    # Wrap dataset to handle padding
    padded_dataset = PaddedGraphDatasetWrapper(dataset)

    generator = torch.Generator().manual_seed(seed)
    primary_train_len = max(1, int(len(padded_dataset) * train_split))
    test_len = max(1, len(padded_dataset) - primary_train_len)
    while primary_train_len + test_len > len(padded_dataset):
        primary_train_len -= 1

    primary_train, test_ds = random_split(
        padded_dataset,
        [primary_train_len, test_len],
        generator=generator,
    )

    val_len = max(1, int(len(primary_train) * val_within_train))
    real_train_len = max(1, len(primary_train) - val_len)
    train_ds, val_ds = random_split(
        primary_train,
        [real_train_len, val_len],
        generator=generator,
    )

    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_mem,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        ),
    )


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    use_amp: bool = True,
    show_progress: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0

    loader_iter = tqdm(loader, desc="Validation", leave=False) if show_progress else loader

    for batch in loader_iter:
        batch = batch.to(device, non_blocking=True)
        y = _safe_y(batch)

        with autocast(
            device_type=device_type,
            enabled=(use_amp and device.type == "cuda"),
        ):
            pred = model(batch).view(-1).float()

            # (Optional) ignore NaN labels if you ever have any
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            loss = loss_fn(pred[mask], y[mask])

        total_loss += float(loss.item()) * int(batch.num_graphs)
        total_graphs += int(batch.num_graphs)

    return total_loss / max(1, total_graphs)


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    lr: list[float]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | None = None,
    loss_type: str = "huber",  # "mse" | "huber"
    huber_delta: float = 1.0,
    grad_clip: float = 5.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.0,
    use_amp: bool = True,
    scheduler: str = "none",  # "none" | "plateau"
    show_progress: bool = True,
    show_val_progress: bool = False,
    log_every_n_batches: int = 20,
    heartbeat_secs: float = 60.0,
    epoch_time_warning_secs: float = 300.0,
) -> tuple[nn.Module, TrainHistory, torch.device]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    if loss_type.lower() == "mse":
        loss_fn: nn.Module = nn.MSELoss()
    elif loss_type.lower() == "huber":
        loss_fn = nn.SmoothL1Loss(beta=huber_delta)
    else:
        raise ValueError("loss_type must be 'mse' or 'huber'")

    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=5,
        )
    else:
        sch = None

    scaler = GradScaler(
        device=device_type,
        enabled=(use_amp and dev.type == "cuda"),
    )

    hist = TrainHistory(train_loss=[], val_loss=[], lr=[])

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"-------- EPOCH : {epoch:03d} --------\n")
        model.train()
        total_loss = 0.0
        total_graphs = 0

        # Prepare progress tracking
        batch_count = 0
        last_heartbeat = time.time()
        train_start_time = time.time()

        # Wrap loader with tqdm if progress is enabled
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            disable=not show_progress,
            file=sys.stdout,
        )

        for batch in train_iter:
            batch = batch.to(dev, non_blocking=True)
            y = _safe_y(batch)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, enabled=(use_amp and dev.type == "cuda")):
                pred = model(batch).view(-1).float()
                mask = torch.isfinite(y)
                if mask.sum() == 0:
                    continue
                loss = loss_fn(pred[mask], y[mask])

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item()) * int(batch.num_graphs)
            total_graphs += int(batch.num_graphs)
            batch_count += 1

            # Update tqdm postfix with running metrics
            if show_progress:
                running_loss = total_loss / max(1, total_graphs)
                train_iter.set_postfix(
                    {
                        "loss": f"{running_loss:.4f}",
                        "graphs": total_graphs,
                    },
                )

            # Periodic structured logging
            if log_every_n_batches > 0 and batch_count % log_every_n_batches == 0:
                running_loss = total_loss / max(1, total_graphs)
                elapsed = time.time() - train_start_time
                batches_per_sec = batch_count / max(elapsed, 0.001)
                remaining_batches = len(train_loader) - batch_count
                eta_secs = remaining_batches / max(batches_per_sec, 0.001)
                logger.debug(
                    f"Epoch {epoch} batch {batch_count}/{len(train_loader)} | "
                    f"loss {running_loss:.6f} | elapsed {elapsed:.1f}s | "
                    f"ETA {eta_secs:.1f}s | {batches_per_sec:.2f} batch/s",
                )

            # Heartbeat logging (wall-clock based)
            if heartbeat_secs > 0:
                now = time.time()
                if now - last_heartbeat >= heartbeat_secs:
                    running_loss = total_loss / max(1, total_graphs)
                    elapsed = time.time() - train_start_time
                    logger.info(
                        f"[Heartbeat] Epoch {epoch} batch {batch_count}/{len(train_loader)} | "
                        f"loss {running_loss:.6f} | elapsed {elapsed:.1f}s | graphs {total_graphs}",
                    )
                    last_heartbeat = now

        train_time = time.time() - train_start_time
        train_loss = total_loss / max(1, total_graphs)

        logger.info(f"Training phase complete ({train_time:.1f}s) | Running validation...")

        val_start_time = time.time()
        val_loss = evaluate_loss(
            model, val_loader, dev, loss_fn, use_amp=use_amp, show_progress=show_val_progress,
        )
        val_time = time.time() - val_start_time

        if sch is not None:
            sch.step(val_loss)

        current_lr = float(opt.param_groups[0]["lr"])
        hist.train_loss.append(float(train_loss))
        hist.val_loss.append(float(val_loss))
        hist.lr.append(current_lr)

        epoch_time = time.time() - epoch_start_time

        logger.info(
            f"Losses | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e} | "
            f"time train={train_time:.1f}s val={val_time:.1f}s total={epoch_time:.1f}s",
        )

        # Warn if epoch is unexpectedly slow
        if epoch_time_warning_secs > 0 and epoch_time > epoch_time_warning_secs:
            logger.warning(
                f"Epoch {epoch} took {epoch_time:.1f}s (>{epoch_time_warning_secs:.0f}s threshold). "
                f"This is expected for large models/datasets.",
            )

        # Early stopping
        if val_loss + early_stopping_min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            logger.debug(f"New best validation loss: {best_val:.6f}")
        else:
            bad_epochs += 1
            logger.debug(f"No improvement: patience {bad_epochs}/{early_stopping_patience}")
            if bad_epochs >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch:03d} | best val {best_val:.6f} | "
                    f"patience exhausted ({bad_epochs}/{early_stopping_patience})",
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist, dev


def plot_training_curves(
    hist: TrainHistory,
    title: str = "Training curves",
    save_fig: bool = False,
    fig_path: str | None = None,
):
    epochs = list(range(1, len(hist.train_loss) + 1))

    plt.figure()
    plt.plot(epochs, hist.train_loss, label="train")
    plt.plot(epochs, hist.val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    # if hist.lr:
    #     plt.figure()
    #     plt.plot(epochs, hist.lr, label="lr")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Learning rate")
    #     plt.title("Learning rate")
    #     plt.grid(True)
    #     plt.show()
    if save_fig and fig_path is not None:
        plt.savefig(fig_path)


def main(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    lr: float = typer.Option(0.001, help="Learning rate"),
    loss_type: str = typer.Option("mse", help="Loss function: mse or huber"),
    show_progress: bool = typer.Option(True, help="Show progress bars during training"),
    show_val_progress: bool = typer.Option(False, help="Show progress bar during validation"),
    log_every_n_batches: int = typer.Option(
        5, help="Log training stats every N batches (0=disable)",
    ),
    heartbeat_secs: float = typer.Option(
        60.0, help="Heartbeat log interval in seconds (0=disable)",
    ),
    epoch_time_warning_secs: float = typer.Option(
        300.0, help="Warn if epoch exceeds N seconds (0=disable)",
    ),
):
    pt_paths = collect_pt_paths("outputs/data")  # or .../samples
    # logger.info(pt_paths[:5])
    # global_feature_variant: str = "binned"
    # node_feature_backend_variant: str | None = None
    # suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    # root = _cache_root_for_paths(pt_paths, suffix=suffix)

    # dataset = QuantumCircuitGraphDataset(
    #     root=root,
    #     pt_paths=pt_paths,
    #     global_feature_variant=global_feature_variant,
    #     node_feature_backend_variant=node_feature_backend_variant,
    # )
    # logger.info(dataset)
    # logger.info(f"Dataset loaded with {len(dataset)} samples.")
    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        pt_paths,
        global_feature_variant="binned",
        batch_size=32,
    )

    node_in_dim = get_node_feature_dim_from_sample(pt_paths)
    global_in_dim = get_global_feature_dim_from_sample(pt_paths)

    model = GNN(
        node_in_dim=node_in_dim,
        global_in_dim=global_in_dim,
        gnn_hidden=32,
        gnn_heads=8,
        global_hidden=16,
        reg_hidden=16,
        num_layers=5,
        dropout_rate=0.1,
    )

    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        huber_delta=1.0,
        early_stopping_patience=15,
        scheduler="plateau",
        show_progress=show_progress,
        show_val_progress=show_val_progress,
        log_every_n_batches=log_every_n_batches,
        heartbeat_secs=heartbeat_secs,
        epoch_time_warning_secs=epoch_time_warning_secs,
    )

    test_loss = evaluate_loss(model, test_loader, dev, nn.MSELoss(), use_amp=True, show_progress=True)
    logger.info(f"Final test loss: {test_loss:.6f}")

    plot_training_curves(
        hist,
        title="GNN SRE regression",
        save_fig=True,
        fig_path="outputs/figures/training_curves.png",
    )

    torch.save(model.state_dict(), "outputs/gnn_model.pt")


def temp_main(
    epochs: int = 15,
    lr: float = 0.001,
    loss_type: str = "mse",
    training_mode: str = "global",     # "global" | "per_family"
    family: str | None = None,
    show_progress: bool = typer.Option(True, help="Show progress bars during training"),
    show_val_progress: bool = typer.Option(False, help="Show progress bar during validation"),
    log_every_n_batches: int = typer.Option(5, help="Log training stats every N batches (0=disable)"),
    heartbeat_secs: float = typer.Option(60.0, help="Heartbeat log interval in seconds (0=disable)"),
    epoch_time_warning_secs: float = typer.Option(300.0, help="Warn if epoch exceeds N seconds (0=disable)"),
):
    data_paths = collect_pt_paths("outputs/data", family=family if training_mode == "per_family" else None)
    if not data_paths:
        logger.error("No data paths found. Check dataset directory and family name.")
        return
    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        data_paths,
        global_feature_variant="binned",
        batch_size=32,
    )

    node_in_dim = get_node_feature_dim_from_sample(data_paths)
    global_in_dim = get_global_feature_dim_from_sample(data_paths)

    model = GNN(
        node_in_dim=node_in_dim,
        global_in_dim=global_in_dim,
        gnn_hidden=32,
        gnn_heads=8,
        global_hidden=16,
        reg_hidden=16,
        num_layers=5,
        dropout_rate=0.1,
    )

    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        huber_delta=1.0,
        early_stopping_patience=15,
        scheduler="plateau",
        show_progress=show_progress,
        show_val_progress=show_val_progress,
        log_every_n_batches=log_every_n_batches,
        heartbeat_secs=heartbeat_secs,
        epoch_time_warning_secs=epoch_time_warning_secs,
    )

    test_loss = evaluate_loss(model, test_loader, dev, nn.MSELoss(), use_amp=True, show_progress=True)
    logger.info(f"Final test loss: {test_loss:.6f}")

    plot_training_curves(
        hist,
        title="GNN SRE regression",
        save_fig=True,
        fig_path="outputs/figures/training_curves.png",
    )

    torch.save(model.state_dict(), "outputs/gnn_model.pt")

if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting GNN training...")
    typer.run(temp_main)
