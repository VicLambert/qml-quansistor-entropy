"""Tests for train_gnn module."""

from __future__ import annotations

import sys

from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_gnn as train_gnn

from qqe.GNN.training.train_config import TrainConfig


class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyDataset:
    def __init__(self):
        self.all_gate_keys = ["x", "y"]


def _make_config(**overrides):
    cfg = TrainConfig(
        epochs=3,
        lr=1e-3,
        loss_type="mse",
        training_mode="global",
        family=None,
        target="sre",
    )
    return replace(cfg, **overrides)


def test_resolve_model_save_path_overwrite(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_text("existing")

    resolved = train_gnn._resolve_model_save_path(str(path), allow_overwrite=True)
    assert resolved == str(path)


def test_resolve_model_save_path_versions(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_text("existing")

    second = tmp_path / "checkpoint_v1.pt"
    second.write_text("existing2")

    resolved = train_gnn._resolve_model_save_path(str(path), allow_overwrite=False)
    assert resolved == str(tmp_path / "checkpoint_v2.pt")


def test_run_training_invalid_mode():
    cfg = _make_config(training_mode="bad-mode")
    with pytest.raises(ValueError, match="training_mode"):
        train_gnn.run_training(cfg)


def test_run_training_invalid_family():
    cfg = _make_config(training_mode="per_family", family="nope")
    with pytest.raises(ValueError, match="Invalid family"):
        train_gnn.run_training(cfg)


def test_run_training_missing_family():
    cfg = _make_config(training_mode="per_family", family=None)
    with pytest.raises(ValueError, match="family must be provided"):
        train_gnn.run_training(cfg)


def test_run_training_no_data_paths(monkeypatch):
    cfg = _make_config()

    monkeypatch.setattr(train_gnn, "collect_files_path", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="No data paths found"):
        train_gnn.run_training(cfg)


def test_run_training_passes_hparams(monkeypatch):
    cfg = _make_config()

    monkeypatch.setattr(train_gnn, "collect_files_path", lambda *_args, **_kwargs: ["x.pt"])

    def fake_build_loaders(*_args, **_kwargs):
        return "train", "val", "test", 5, 7, DummyDataset()

    monkeypatch.setattr(train_gnn, "build_loaders", fake_build_loaders)

    def fake_train_model(model, *_args, **kwargs):
        assert isinstance(model, DummyModel)
        assert kwargs["weight_decay"] == 0.2
        assert kwargs["grad_clip"] == 1.5
        assert kwargs["early_stopping_patience"] == 4
        assert kwargs["early_stopping_min_delta"] == 0.01
        return model, "hist", "cpu"

    monkeypatch.setattr(train_gnn, "train_model", fake_train_model)
    monkeypatch.setattr(train_gnn, "build_loss", lambda *_args, **_kwargs: "loss")
    monkeypatch.setattr(train_gnn, "evaluate_loss", lambda *_args, **_kwargs: 0.5)

    monkeypatch.setattr(train_gnn, "GNN", DummyModel)

    model, hist, test_loss, node_in_dim, global_in_dim, base_dataset, model_hparams = (
        train_gnn.run_training(
            cfg,
            model_hparams={"gnn_hidden": 64},
            train_hparams={
                "weight_decay": 0.2,
                "grad_clip": 1.5,
                "early_stopping_patience": 4,
                "early_stopping_min_delta": 0.01,
            },
        )
    )

    assert isinstance(model, DummyModel)
    assert hist == "hist"
    assert test_loss == 0.5
    assert node_in_dim == 5
    assert global_in_dim == 7
    assert isinstance(base_dataset, DummyDataset)
    assert model_hparams["gnn_hidden"] == 64


def test_run_training_nn_unsupported_model_type():
    cfg = _make_config()
    with pytest.raises(ValueError, match="Unsupported model type"):
        train_gnn.run_training_NN(cfg, model_type="nope")


def test_run_training_nn_models(monkeypatch):
    cfg = _make_config()

    monkeypatch.setattr(train_gnn, "collect_files_path", lambda *_args, **_kwargs: ["x.pt"])

    def fake_build_loaders_nn(*_args, **_kwargs):
        return "train", "val", "test", 6, DummyDataset()

    monkeypatch.setattr(train_gnn, "build_loaders_NN", fake_build_loaders_nn)

    def fake_train_model(model, *_args, **_kwargs):
        return model, "hist", "cpu"

    monkeypatch.setattr(train_gnn, "train_model", fake_train_model)
    monkeypatch.setattr(train_gnn, "build_loss", lambda *_args, **_kwargs: "loss")
    monkeypatch.setattr(train_gnn, "evaluate_loss", lambda *_args, **_kwargs: 0.3)

    monkeypatch.setattr(train_gnn, "NN", DummyModel)
    monkeypatch.setattr(train_gnn, "Regressor", DummyModel)

    model, hist, test_loss, global_in_dim, base_dataset = train_gnn.run_training_NN(
        cfg,
        model_type="MLP",
        model_params={"hidden_dim": 128, "dropout_rate": 0.2},
    )

    assert isinstance(model, DummyModel)
    assert hist == "hist"
    assert test_loss == 0.3
    assert global_in_dim == 6
    assert isinstance(base_dataset, DummyDataset)

    model, _, _, _, _ = train_gnn.run_training_NN(
        cfg,
        model_type="regressor",
        model_params={"hidden_dim": 64, "dropout_rate": 0.0},
    )

    assert isinstance(model, DummyModel)
