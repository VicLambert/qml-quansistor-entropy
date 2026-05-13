from __future__ import annotations

import logging

import torch
import typer

from qqe.src.experiments.plotting import (
    plot_fixed_layers_vary_qubits,
    plot_fixed_qubits_vary_layers,
)
from qqe.src.GNN.prediction.datasets import (
    build_loader,
    build_prediction_dataset,
    collect_prediction_paths,
)
from qqe.src.GNN.prediction.inference import predict
from qqe.src.GNN.prediction.model import build_model, checkpoint_path, load_checkpoint
from qqe.src.GNN.prediction.utils import save_predictions_csv
from qqe.src.GNN.training.utils import collect_dataset_paths
from qqe.src.utils import configure_logger

logger = logging.getLogger(__name__)

def main(
    model_path: str = typer.Option(" models/nn_model_huber_global.pt", help="Path to model checkpoint."),
    model_kind: str = typer.Option("gnn", help="Model type: 'gnn' or 'nn'."),
    training_scope: str = typer.Option("family", help="'global' or 'family'."),
    loss_type: str = typer.Option("huber", help="Loss type used during training, e.g. 'mse' or 'huber'."),
    model_family: str | None = typer.Option("haar", help="Family used if training_scope='family'."),
    dataset_root: str = typer.Option("outputs/data", help="Root folder containing prediction files."),
    dataset_family: str | None = typer.Option("haar", help="Optional family to predict on."),
    batch_size: int = typer.Option(32, help="Batch size."),
    global_feature_variant: str = typer.Option("binned", help="Global feature variant."),
    node_feature_backend_variant: str | None = typer.Option(None, help="Optional node feature backend variant."),
    plot_n_layers: int | None = typer.Option(80, help="Make plot at fixed n_layers, varying n_qubits."),
    plot_n_qubits: int | None = typer.Option(16, help="Make plot at fixed n_qubits, varying n_layers."),
    split_by_family: bool = typer.Option(True, help="Plot separate curves for each family."),
    show_progress: bool = typer.Option(True, help="Show progress bar during prediction."),
):
    ckpt_path = checkpoint_path(model_kind, training_scope, model_family, loss_type)
    logger.info("Loading checkpoint: %s", model_path)
    output_csv = f"../outputs/predictions/{training_scope}/{model_kind}_predictions_{model_family or 'global'}.csv"

    state_dict, model_config, feature_config = load_checkpoint(model_path)

    model = build_model(model_kind, model_config)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # pt_paths = collect_prediction_paths(dataset_root, dataset_family)
    prediction_paths = collect_dataset_paths(
        dataset_root,
        family=dataset_family,
        split="all",
    )
    if not prediction_paths:
        raise RuntimeError("No prediction .pt files found.")

    logger.info("Found %d prediction files", len(prediction_paths))

    dataset = build_prediction_dataset(
        prediction_paths,
        global_feature_variant=feature_config.get("global_feature_variant", global_feature_variant),
        node_feature_backend_variant=feature_config.get("node_feature_backend_variant", node_feature_backend_variant),
        fixed_all_gate_keys=feature_config.get("all_gate_keys"),
    )

    loader = build_loader(
        model_kind,
        dataset,
        batch_size=batch_size,
        target_node_dim=model_config.get("node_in_dim"),
        target_global_dim=model_config.get("global_in_dim"),
    )

    rows = predict(
        model,
        loader,
        model_kind=model_kind,
        device=device,
        show_progress=show_progress,
    )
    save_predictions_csv(rows, output_csv)

    logger.info("Saved %d predictions to %s", len(rows), output_csv)

    if plot_n_layers is not None:
        plot_path = f"../outputs/figures/predictions/{training_scope}/{model_kind}_pred_layers_{model_family or 'global'}.png"
        plot_fixed_layers_vary_qubits(
            rows,
            n_layers=plot_n_layers,
            output_path=plot_path,
            split_by_family=split_by_family,
        )
        logger.info("Saved fixed-layer plot to %s", plot_path)

    if plot_n_qubits is not None:
        plot_path = f"../outputs/figures/predictions/{training_scope}/{model_kind}_pred_qubits_{model_family or 'global'}.png"
        plot_fixed_qubits_vary_layers(
            rows,
            n_qubits=plot_n_qubits,
            output_path=plot_path,
            split_by_family=split_by_family,
        )
        logger.info("Saved fixed-qubit plot to %s", plot_path)


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting prediction...")
    typer.run(main)
