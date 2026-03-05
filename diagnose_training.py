"""Training diagnostic script to identify bottlenecks.
"""
import hashlib
import os
import time

from pathlib import Path

import torch

from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader

# Reproduce the imports and setup from notebook
from qqe.GNN.physics_aware_NN import GNN


class QuantumCircuitGraphDataset(PyGDataset):
    def __init__(
        self,
        root: str,
        pt_paths: list[str],
        global_feature_variant: str = "baseline",
        node_feature_backend_variant: str | None = None,
        transform=None,
        pre_transform=None,
    ):
        self.pt_paths = [str(p) for p in pt_paths]
        self.global_feature_variant = global_feature_variant
        self.node_feature_backend_variant = node_feature_backend_variant
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.pt_paths)

    def get(self, idx: int):
        obj = torch.load(self.pt_paths[idx], map_location="cpu")
        x = obj["x"]
        edge_index = obj["edge_index"]
        g = obj["global_features"]
        y_val = obj.get("sre", None)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)
        if g.dtype != torch.float32:
            g = g.to(torch.float32)

        if y_val is None:
            y = torch.tensor([float("nan")], dtype=torch.float32)
        else:
            y = torch.tensor([float(y_val)], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.global_features = g
        data.num_qubits = int(obj.get("meta", {}).get("n_qubits", 0))
        data.gate_counts = obj.get("gate_counts", {})
        data.meta = obj.get("meta", {})
        return data


def collect_pt_paths(dataset_dir: str) -> list[str]:
    d = Path(dataset_dir)
    paths = sorted((d / "encoding_data_quimb_fwht").glob("*.pt"))
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


def build_train_val_test_loaders(
    pt_paths: list[str],
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    num_workers: int = 0,
):
    suffix = f"{global_feature_variant}_backend_none"
    root = _cache_root_for_paths(pt_paths, suffix=suffix)

    dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
    )

    generator = torch.Generator().manual_seed(seed)
    primary_train_len = max(1, int(len(dataset) * train_split))
    test_len = max(1, len(dataset) - primary_train_len)

    primary_train, test_ds = random_split(dataset, [primary_train_len, test_len], generator=generator)

    val_len = max(1, int(len(primary_train) * val_within_train))
    real_train_len = max(1, len(primary_train) - val_len)
    train_ds, val_ds = random_split(primary_train, [real_train_len, val_len], generator=generator)

    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem),
    )


def main():
    print("=" * 60)
    print("TRAINING DIAGNOSTICS")
    print("=" * 60)

    # Setup
    pt_paths = collect_pt_paths("qqe/data/")
    print(f"\n📊 Dataset: {len(pt_paths)} samples")

    # Check sample data
    obj = torch.load(pt_paths[0], map_location="cpu")
    node_in_dim = int(obj["x"].shape[1])
    global_in_dim = int(obj["global_features"].numel())
    print(f"   Node features: {node_in_dim}")
    print(f"   Global features: {global_in_dim}")

    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Test different num_workers
    print("\n⏱️  Testing DataLoader Performance...")
    for num_workers in [0, 2]:
        train_loader, val_loader, test_loader = build_train_val_test_loaders(
            pt_paths,
            global_feature_variant="binned",
            batch_size=32,
            num_workers=num_workers,
        )

        # Time data loading
        start = time.time()
        for i, batch in enumerate(train_loader):
            if i >= 10:  # Just first 10 batches
                break
        elapsed = time.time() - start
        print(f"   num_workers={num_workers}: {elapsed:.2f}s for 10 batches ({elapsed/10:.3f}s/batch)")

    # Use best setting
    train_loader, val_loader, test_loader = build_train_val_test_loaders(
        pt_paths,
        global_feature_variant="binned",
        batch_size=32,
        num_workers=2,
    )

    print(f"\n   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Create model
    model = GNN(
        node_in_dim=node_in_dim,
        global_in_dim=global_in_dim,
        gnn_hidden=32,
        gnn_heads=8,
        global_hidden=16,
        reg_hidden=16,
        num_layers=5,
        dropout_rate=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")

    # Test forward pass timing
    print("\n⚡ Forward Pass Timing...")
    batch = next(iter(train_loader)).to(device)

    # Warmup
    for _ in range(3):
        _ = model(batch)

    # Time it
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start = time.time()
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"   Average forward pass: {avg_time*1000:.2f}ms")
    print(f"   Throughput: ~{1/avg_time:.1f} batches/sec")

    # Simulate one full epoch
    print("\n🏃 Simulating One Epoch...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    epoch_start = time.time()
    data_time = 0
    forward_time = 0
    backward_time = 0

    batch_iter_start = time.time()
    for i, batch in enumerate(train_loader):
        data_time += time.time() - batch_iter_start

        batch = batch.to(device, non_blocking=True)
        y = batch.y.float().view(-1)

        optimizer.zero_grad(set_to_none=True)

        fwd_start = time.time()
        pred = model(batch).view(-1)
        loss = loss_fn(pred, y)
        forward_time += time.time() - fwd_start

        bwd_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time += time.time() - bwd_start

        if i >= 49:  # First 50 batches
            break

        batch_iter_start = time.time()

    total_epoch = time.time() - epoch_start

    print(f"   Total time (50 batches): {total_epoch:.2f}s")
    print(f"   Data loading: {data_time:.2f}s ({data_time/total_epoch*100:.1f}%)")
    print(f"   Forward pass: {forward_time:.2f}s ({forward_time/total_epoch*100:.1f}%)")
    print(f"   Backward pass: {backward_time:.2f}s ({backward_time/total_epoch*100:.1f}%)")
    print(f"   Time/batch: {total_epoch/50:.3f}s")

    # Estimate full training time
    num_batches = len(train_loader)
    estimated_epoch_time = (total_epoch / 50) * num_batches
    print("\n📈 Full Epoch Estimate:")
    print(f"   {num_batches} batches × {total_epoch/50:.3f}s = {estimated_epoch_time:.1f}s (~{estimated_epoch_time/60:.1f} min)")
    print(f"   For 10 epochs: ~{estimated_epoch_time*10/60:.1f} minutes")
    print(f"   For 200 epochs: ~{estimated_epoch_time*200/60:.1f} minutes")

    # Bottleneck analysis
    print("\n🔍 Bottleneck Analysis:")
    if data_time / total_epoch > 0.3:
        print(f"   ⚠️  Data loading is slow ({data_time/total_epoch*100:.1f}%)")
        print(f"       Try: num_workers={min(os.cpu_count() or 4, 4)}, persistent_workers=True")
    if forward_time / total_epoch > 0.5:
        print(f"   ⚠️  Forward pass is dominant ({forward_time/total_epoch*100:.1f}%)")
        print("       Model is compute-bound - expected for GNNs")

    print("\n✅ Diagnosis complete!")


if __name__ == "__main__":
    main()
