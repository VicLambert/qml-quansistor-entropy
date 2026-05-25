from __future__ import annotations

import logging


import optuna


from GNN.physics_aware_NN import GNN, NN
from GNN.training.datasets import make_loaders
from GNN.training.train import train_model

logger = logging.getLogger(__name__)

def objective_GNN(trial: optuna.Trial, prepared_data, epochs=10):
    batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_loader, val_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=batch_size,
    )
    train_hparams = {
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-2, log=True,
        ),
        "grad_clip": trial.suggest_float(
            "grad_clip", 1e-2, 1e2, log=True,
        ),
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-5,
    }

    model_hparams = {
        "gnn_hidden": trial.suggest_categorical("gnn_hidden", [16, 32, 64, 128]),
        "gnn_heads": trial.suggest_categorical("gnn_heads", [2, 4, 8]),
        "global_hidden": trial.suggest_categorical("global_hidden", [16, 32, 64, 128]),
        "reg_hidden": trial.suggest_categorical("reg_hidden", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }
    model = GNN(
        node_in_dim=prepared_data.node_in_dim,
        global_in_dim=prepared_data.global_in_dim,
        gnn_hidden=model_hparams["gnn_hidden"],
        gnn_heads=model_hparams["gnn_heads"],
        global_hidden=model_hparams["global_hidden"],
        reg_hidden=model_hparams["reg_hidden"],
        num_layers=model_hparams["num_layers"],
        dropout_rate=model_hparams["dropout_rate"],
    )

    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        loss_type=trial.suggest_categorical("loss_type", ["mse", "huber"]),
        scheduler="plateau",
        show_progress=False,
        show_val_progress=False,
        weight_decay=train_hparams["weight_decay"],
        grad_clip=train_hparams["grad_clip"],
        early_stopping_patience=train_hparams["early_stopping_patience"],
        early_stopping_min_delta=train_hparams["early_stopping_min_delta"],
    )

    return min(hist.val_loss)

def objective_NN(trial: optuna.Trial, prepared_data, epochs=10):
    batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_loader, val_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=batch_size,
    )

    train_hparams = {
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-2, log=True,
        ),
        "grad_clip": trial.suggest_float(
            "grad_clip", 1e-2, 1e2, log=True,
        ),
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-5,
    }

    model_hparams = {
        "nn_hidden": trial.suggest_categorical("nn_hidden", [32, 64, 128, 256]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }
    model = NN(
        in_dim=prepared_data.global_in_dim,
        hidden_dim=model_hparams["nn_hidden"],
        dropout_rate=model_hparams["dropout_rate"],
    )

    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        loss_type=trial.suggest_categorical("loss_type", ["mse", "huber"]),
        scheduler="plateau",
        show_progress=False,
        show_val_progress=False,
        weight_decay=train_hparams["weight_decay"],
        grad_clip=train_hparams["grad_clip"],
        early_stopping_patience=train_hparams["early_stopping_patience"],
        early_stopping_min_delta=train_hparams["early_stopping_min_delta"],
    )

    return min(hist.val_loss)
