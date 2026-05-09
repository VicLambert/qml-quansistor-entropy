"""Tests for train_gnn module."""

from __future__ import annotations

import sys

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qqe.GNN.training import runners
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

    resolved = runners._resolve_model_save_path(str(path), allow_overwrite=True)
    assert resolved == str(path)


def test_resolve_model_save_path_versions(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_text("existing")

    second = tmp_path / "checkpoint_v1.pt"
    second.write_text("existing2")

    resolved = runners._resolve_model_save_path(str(path), allow_overwrite=False)
    assert resolved == str(tmp_path / "checkpoint_v2.pt")


def test_train_invalid_mode():
    with pytest.raises(ValueError, match="training_mode"):
        runners.train(training_mode="bad-mode", epochs=1)


def test_train_invalid_family():
    with pytest.raises(ValueError, match="Invalid family"):
        runners.train(training_mode="per_family", family="nope", epochs=1)


def test_train_missing_family():
    with pytest.raises(ValueError, match="family must be provided"):
        runners.train(training_mode="per_family", family=None, epochs=1)


def test_train_no_data_paths():
    with patch("qqe.GNN.training.runners.collect_files_path", return_value=[]):
        with pytest.raises(RuntimeError, match="No data paths found"):
            runners.train(epochs=1)


@patch("qqe.GNN.training.runners.collect_files_path")
@patch("qqe.GNN.training.runners.build_loaders")
@patch("qqe.GNN.training.runners.train_model")
@patch("qqe.GNN.training.runners.build_loss")
@patch("qqe.GNN.training.runners.evaluate_loss")
@patch("qqe.GNN.training.runners.plot_training_curves")
@patch("qqe.GNN.training.runners.GNN")
def test_train_gnn_with_hparams(
    mock_gnn,
    mock_plot,
    mock_eval,
    mock_loss,
    mock_train,
    mock_loaders,
    mock_collect,
):
    """Test training GNN model with custom hyperparameters."""
    mock_collect.return_value = ["x.pt"]
    mock_loaders.return_value = ("train", "val", "test", 5, 7, DummyDataset())

    mock_train_history = MagicMock()
    mock_train_history.train_loss = [0.5]
    mock_train_history.val_loss = [0.4]
    mock_train_history.lr = [0.001]
    mock_train.return_value = (DummyModel(), mock_train_history, "cpu")
    mock_eval.return_value = 0.5
    mock_gnn.return_value = DummyModel()

    model, test_loss, hist, checkpoint = runners.train(
        model_type="gnn",
        epochs=1,
        model_hparams={"gnn_hidden": 64},
        train_hparams={
            "weight_decay": 0.2,
            "grad_clip": 1.5,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.01,
        },
    )

    assert isinstance(model, DummyModel)
    assert isinstance(test_loss, float)
    assert hist is not None
    assert isinstance(checkpoint, dict)


@patch("qqe.GNN.training.runners.collect_files_path")
@patch("qqe.GNN.training.runners.build_loaders")
@patch("qqe.GNN.training.runners.train_model")
@patch("qqe.GNN.training.runners.build_loss")
@patch("qqe.GNN.training.runners.evaluate_loss")
@patch("qqe.GNN.training.runners.plot_training_curves")
@patch("qqe.GNN.training.runners.NN")
def test_train_nn_model(
    mock_nn,
    mock_plot,
    mock_eval,
    mock_loss,
    mock_train,
    mock_loaders,
    mock_collect,
):
    """Test training NN model."""
    mock_collect.return_value = ["x.pt"]
    mock_loaders.return_value = ("train", "val", "test", 6, DummyDataset())

    mock_train_history = MagicMock()
    mock_train_history.train_loss = [0.5]
    mock_train_history.val_loss = [0.4]
    mock_train_history.lr = [0.001]
    mock_train.return_value = (DummyModel(), mock_train_history, "cpu")
    mock_eval.return_value = 0.3
    mock_nn.return_value = DummyModel()

    model, test_loss, hist, checkpoint = runners.train(
        model_type="nn",
        epochs=1,
        model_hparams={"hidden_dim": 128, "dropout_rate": 0.2},
    )

    assert isinstance(model, DummyModel)
    assert test_loss == 0.3
    assert hist is not None


@patch("qqe.GNN.training.runners.collect_files_path")
@patch("qqe.GNN.training.runners.build_loaders")
@patch("qqe.GNN.training.runners.train_model")
@patch("qqe.GNN.training.runners.build_loss")
@patch("qqe.GNN.training.runners.evaluate_loss")
@patch("qqe.GNN.training.runners.plot_training_curves")
@patch("qqe.GNN.training.runners.Regressor")
def test_train_regressor_model(
    mock_reg,
    mock_plot,
    mock_eval,
    mock_loss,
    mock_train,
    mock_loaders,
    mock_collect,
):
    """Test training Regressor model."""
    mock_collect.return_value = ["x.pt"]
    mock_loaders.return_value = ("train", "val", "test", 6, DummyDataset())

    mock_train_history = MagicMock()
    mock_train.return_value = (DummyModel(), mock_train_history, "cpu")
    mock_eval.return_value = 0.3
    mock_reg.return_value = DummyModel()

    model, test_loss, hist, checkpoint = runners.train(
        model_type="regressor",
        epochs=1,
        model_hparams={"hidden_dim": 64, "dropout_rate": 0.0},
    )

    assert isinstance(model, DummyModel)
    assert test_loss == 0.3


def test_model_registry_structure():
    """Test MODEL_REGISTRY has expected structure."""
    assert "gnn" in runners.MODEL_REGISTRY
    assert "nn" in runners.MODEL_REGISTRY
    assert "regressor" in runners.MODEL_REGISTRY

    for model_type, spec in runners.MODEL_REGISTRY.items():
        assert "build_loaders" in spec
        assert "build_model" in spec
        assert "returns_nodes_dim" in spec
