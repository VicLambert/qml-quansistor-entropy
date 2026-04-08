from __future__ import annotations

import csv
import itertools
import json
import logging

from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

import torch
import typer

from qqe.GNN.training.train_config import TrainConfig
from qqe.utils import configure_logger
from train_gnn import run_training

logger = logging.getLogger(__name__)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main(
    epochs: int = 10,
    training_mode: str = "global",  # "global" | "per_family"
    family: str | None = None,
    loss_type: str = "huber",  # "mse" | "huber" | "l1"
    batch_size: int = 32,
    seed: int = 42,
    show_progress: bool = typer.Option(default=True, help="Show training progress bars"),
    show_val_progress: bool = typer.Option(
        default=False, help="Show validation progress bars"
    ),
    gnn_hidden_grid: str = typer.Option("16,32,64", help="Comma-separated values"),
    gnn_heads_grid: str = typer.Option("2,4,8", help="Comma-separated values"),
    global_hidden_grid: str = typer.Option("16,32", help="Comma-separated values"),
    reg_hidden_grid: str = typer.Option("16,32", help="Comma-separated values"),
    num_layers_grid: str = typer.Option("3,5,7", help="Comma-separated values"),
    dropout_grid: str = typer.Option("0.0,0.1,0.2", help="Comma-separated values"),
    lr_grid: str = typer.Option("0.001,0.0005", help="Comma-separated values"),
    weight_decay_grid: str = typer.Option("0.0,0.0001", help="Comma-separated values"),
    max_trials: int = typer.Option(20, help="Cap number of trials (0 = all combinations)"),
    results_dir: str = typer.Option(
        "outputs/runs/hparam_search", help="Directory for search outputs"
    ),
    show_grid_progress: bool = typer.Option(
        default=True, help="Show grid search progress",
    ),
):
    train_cfg = TrainConfig(
        epochs=epochs,
        lr=1e-3,
        loss_type=loss_type,
        batch_size=batch_size,
        training_mode=training_mode,
        family=family,
        seed=seed,
        show_progress=show_progress,
        show_val_progress=show_val_progress,
        log_batch_loss_every=0,
    )

    grid = {
        "gnn_hidden": _parse_int_list(gnn_hidden_grid),
        "gnn_heads": _parse_int_list(gnn_heads_grid),
        "global_hidden": _parse_int_list(global_hidden_grid),
        "reg_hidden": _parse_int_list(reg_hidden_grid),
        "num_layers": _parse_int_list(num_layers_grid),
        "dropout_rate": _parse_float_list(dropout_grid),
        "lr": _parse_float_list(lr_grid),
        "weight_decay": _parse_float_list(weight_decay_grid),
    }

    keys = list(grid.keys())
    values_product = list(itertools.product(*(grid[k] for k in keys)))

    if max_trials > 0:
        values_product = values_product[:max_trials]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    best_trial: dict | None = None
    best_val_loss = float("inf")
    best_checkpoint: dict | None = None

    logger.info("Starting grid search with %d trial(s)", len(values_product))

    with tqdm(
        total=len(values_product),
        desc="Grid Search Trials",
        unit="trial",
        disable=not show_grid_progress,
    ) as pbar:
        for idx, combo in enumerate(values_product, start=1):
            params = dict(zip(keys, combo, strict=True))
            model_hparams = {
                "gnn_hidden": int(params["gnn_hidden"]),
                "gnn_heads": int(params["gnn_heads"]),
                "global_hidden": int(params["global_hidden"]),
                "reg_hidden": int(params["reg_hidden"]),
                "num_layers": int(params["num_layers"]),
                "dropout_rate": float(params["dropout_rate"]),
            }
            train_hparams = {
                "weight_decay": float(params["weight_decay"]),
            }
            trial_cfg = TrainConfig(**asdict(train_cfg))
            trial_cfg.lr = float(params["lr"])

            logger.info("Trial %d/%d params=%s", idx, len(values_product), params)

            (
                model,
                hist,
                test_loss,
                node_in_dim,
                global_in_dim,
                base_dataset,
                used_model_hparams,
            ) = run_training(
                trial_cfg,
                model_hparams=model_hparams,
                train_hparams=train_hparams,
            )

            best_trial_val = min(hist.val_loss) if hist.val_loss else float("inf")
            final_val = hist.val_loss[-1] if hist.val_loss else float("inf")

            trial_result = {
                "trial": idx,
                "best_val_loss": float(best_trial_val),
                "final_val_loss": float(final_val),
                "test_loss": float(test_loss),
                "lr": float(trial_cfg.lr),
                "weight_decay": float(train_hparams["weight_decay"]),
                **used_model_hparams,
            }
            results.append(trial_result)

            if best_trial_val < best_val_loss:
                best_val_loss = best_trial_val
                best_trial = trial_result
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "node_in_dim": node_in_dim,
                        "global_in_dim": global_in_dim,
                        **used_model_hparams,
                    },
                    "train_config": asdict(trial_cfg),
                    "feature_config": {
                        "global_feature_variant": trial_cfg.global_feature_variant,
                        "node_feature_backend_variant": trial_cfg.node_feature_backend_variant,
                        "all_gate_keys": getattr(base_dataset, "all_gate_keys", None),
                        "family_projection": (
                            trial_cfg.family if trial_cfg.training_mode == "per_family" else None
                        ),
                    },
                    "final_metrics": {
                        "best_val_loss": float(best_trial_val),
                        "test_loss": float(test_loss),
                    },
                }
            if show_grid_progress:
                pbar.set_postfix(
                    trial=idx,
                    val=f"{best_trial_val:.4f}",
                    best=f"{best_val_loss:.4f}",
                )
                pbar.update(1)

    results = sorted(results, key=lambda x: x["best_val_loss"])

    json_path = out_dir / "grid_search_results.json"
    csv_path = out_dir / "grid_search_results.csv"
    best_model_path = out_dir / "best_gnn_model_from_grid.pt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "search_space": grid,
                "num_trials": len(values_product),
                "best_trial": best_trial,
                "results": results,
            },
            f,
            indent=2,
        )

    if results:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    if best_checkpoint is not None:
        torch.save(best_checkpoint, best_model_path)

    logger.info("Grid search complete.")
    logger.info("Best trial: %s", best_trial)
    logger.info("Saved ranked results to %s and %s", json_path, csv_path)
    if best_checkpoint is not None:
        logger.info("Saved best model checkpoint to %s", best_model_path)


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    typer.run(main)
