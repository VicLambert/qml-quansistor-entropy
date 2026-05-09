"""Tests for qqe.GNN.training.runners module."""

from __future__ import annotations

import sys
import tempfile

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qqe.GNN.training.runners import (
    MODEL_REGISTRY,
    _resolve_model_save_path,
    train,
)


class TestResolveModelSavePath:
    """Tests for _resolve_model_save_path function."""

    def test_resolve_new_path(self):
        """Test resolving path for new model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pt")
            resolved = _resolve_model_save_path(path, allow_overwrite=False)

            assert resolved == path

    def test_resolve_existing_path_overwrite(self):
        """Test overwriting existing model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            path.write_text("existing")

            resolved = _resolve_model_save_path(str(path), allow_overwrite=True)

            assert resolved == str(path)

    def test_resolve_existing_path_versioning(self):
        """Test versioning when file exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            path.write_text("existing")

            resolved = _resolve_model_save_path(str(path), allow_overwrite=False)

            assert "v1" in resolved
            assert resolved != str(path)

    def test_resolve_multiple_versions(self):
        """Test versioning with multiple existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "model.pt"
            base_path.write_text("existing")

            v1_path = Path(tmpdir) / "model_v1.pt"
            v1_path.write_text("existing_v1")

            resolved = _resolve_model_save_path(str(base_path), allow_overwrite=False)

            assert "v2" in resolved

    def test_resolve_preserves_extension(self):
        """Test that file extension is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            path.write_text("existing")

            resolved = _resolve_model_save_path(str(path), allow_overwrite=False)

            assert resolved.endswith(".pt")


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_registry_has_expected_models(self):
        """Test registry contains expected model types."""
        expected_models = {"gnn", "nn", "regressor"}
        assert set(MODEL_REGISTRY.keys()) == expected_models

    def test_gnn_registry_entry(self):
        """Test GNN registry entry structure."""
        gnn_spec = MODEL_REGISTRY["gnn"]

        assert "build_loaders" in gnn_spec
        assert "build_model" in gnn_spec
        assert "returns_nodes_dim" in gnn_spec
        assert "objective_fn" in gnn_spec
        assert gnn_spec["returns_nodes_dim"] is True

    def test_nn_registry_entry(self):
        """Test NN registry entry structure."""
        nn_spec = MODEL_REGISTRY["nn"]

        assert "build_loaders" in nn_spec
        assert "build_model" in nn_spec
        assert "returns_nodes_dim" in nn_spec
        assert "objective_fn" in nn_spec
        assert nn_spec["returns_nodes_dim"] is False

    def test_regressor_registry_entry(self):
        """Test Regressor registry entry structure."""
        reg_spec = MODEL_REGISTRY["regressor"]

        assert "build_loaders" in reg_spec
        assert "build_model" in reg_spec
        assert "returns_nodes_dim" in reg_spec
        assert reg_spec["returns_nodes_dim"] is False

    def test_build_model_callables(self):
        """Test that build_model functions are callable."""
        for model_type, spec in MODEL_REGISTRY.items():
            assert callable(spec["build_model"]), f"{model_type} build_model is not callable"

    def test_build_loaders_callables(self):
        """Test that build_loaders functions are callable."""
        for model_type, spec in MODEL_REGISTRY.items():
            assert callable(
                spec["build_loaders"]
            ), f"{model_type} build_loaders is not callable"


class TestTrain:
    """Tests for train function."""

    @pytest.fixture
    def mock_dependencies(self):
        """Setup common mocks for train function."""
        with (
            patch("qqe.GNN.training.runners.collect_files_path") as mock_collect,
            patch("qqe.GNN.training.runners.build_loaders") as mock_loaders_gnn,
            patch("qqe.GNN.training.runners.build_loaders_NN") as mock_loaders_nn,
            patch("qqe.GNN.training.runners.train_model") as mock_train,
            patch("qqe.GNN.training.runners.build_loss") as mock_loss,
            patch("qqe.GNN.training.runners.evaluate_loss") as mock_eval,
            patch("qqe.GNN.training.runners.plot_training_curves") as mock_plot,
            patch("qqe.GNN.training.runners.GNN") as mock_gnn,
            patch("qqe.GNN.training.runners.NN") as mock_nn,
            patch("qqe.GNN.training.runners.Regressor") as mock_reg,
        ):

            # Setup return values
            mock_collect.return_value = ["data1.pt"]
            mock_loaders_gnn.return_value = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
                8,
                64,
                MagicMock(),
            )
            mock_loaders_nn.return_value = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
                64,
                MagicMock(),
            )

            mock_train_history = MagicMock()
            mock_train_history.train_loss = [0.5]
            mock_train_history.val_loss = [0.4]
            mock_train_history.lr = [0.001]

            mock_train.return_value = (MagicMock(), mock_train_history, torch.device("cpu"))
            mock_loss.return_value = nn.MSELoss()
            mock_eval.return_value = 0.3

            mock_gnn.return_value = MagicMock(spec=nn.Module)
            mock_nn.return_value = MagicMock(spec=nn.Module)
            mock_reg.return_value = MagicMock(spec=nn.Module)

            yield {
                "collect": mock_collect,
                "loaders_gnn": mock_loaders_gnn,
                "loaders_nn": mock_loaders_nn,
                "train": mock_train,
                "loss": mock_loss,
                "eval": mock_eval,
                "plot": mock_plot,
                "gnn": mock_gnn,
                "nn": mock_nn,
                "reg": mock_reg,
            }

    def test_train_invalid_model_type(self, mock_dependencies):
        """Test error on invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            train(model_type="invalid", epochs=1)

    def test_train_invalid_training_mode(self, mock_dependencies):
        """Test error on invalid training mode."""
        with pytest.raises(ValueError, match="training_mode must be"):
            train(training_mode="invalid", epochs=1)

    def test_train_per_family_missing_family(self, mock_dependencies):
        """Test error when per_family mode without family."""
        with pytest.raises(ValueError, match="family must be provided"):
            train(training_mode="per_family", epochs=1)

    def test_train_per_family_invalid_family(self, mock_dependencies):
        """Test error on invalid family in per_family mode."""
        with pytest.raises(ValueError, match="Invalid family"):
            train(training_mode="per_family", family="invalid", epochs=1)

    def test_train_no_data_paths(self, mock_dependencies):
        """Test error when no data paths found."""
        mock_dependencies["collect"].return_value = []

        with pytest.raises(RuntimeError, match="No data paths found"):
            train(epochs=1)

    def test_train_gnn_model_global_mode(self, mock_dependencies):
        """Test training GNN model in global mode."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            training_mode="global",
        )

        assert model is not None
        assert isinstance(test_loss, float)
        assert isinstance(hist, MagicMock)
        assert isinstance(checkpoint, dict)

    def test_train_nn_model_global_mode(self, mock_dependencies):
        """Test training NN model in global mode."""
        model, test_loss, hist, checkpoint = train(
            model_type="nn",
            epochs=1,
            training_mode="global",
        )

        assert model is not None
        assert isinstance(test_loss, float)

    def test_train_regressor_model(self, mock_dependencies):
        """Test training Regressor model."""
        model, test_loss, hist, checkpoint = train(
            model_type="regressor",
            epochs=1,
            training_mode="global",
        )

        assert model is not None

    def test_train_per_family_mode(self, mock_dependencies):
        """Test training in per_family mode."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            training_mode="per_family",
            family="random",
        )

        assert model is not None

    def test_train_all_families(self, mock_dependencies):
        """Test training with each valid family."""
        valid_families = {"haar", "clifford", "quansistor", "random"}

        for family in valid_families:
            model, test_loss, hist, checkpoint = train(
                model_type="gnn",
                epochs=1,
                training_mode="per_family",
                family=family,
            )

            assert model is not None

    def test_train_custom_hparams(self, mock_dependencies):
        """Test training with custom hyperparameters."""
        model_hparams = {
            "gnn_hidden": 64,
            "gnn_heads": 4,
            "global_hidden": 32,
            "reg_hidden": 32,
            "num_layers": 3,
            "dropout_rate": 0.2,
        }

        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            model_hparams=model_hparams,
        )

        assert model is not None

    def test_train_custom_loss_type(self, mock_dependencies):
        """Test training with different loss types."""
        for loss_type in ["mse", "huber"]:
            model, test_loss, hist, checkpoint = train(
                model_type="gnn",
                epochs=1,
                loss_type=loss_type,
            )

            assert model is not None

    def test_train_checkpoint_structure(self, mock_dependencies):
        """Test checkpoint has expected structure."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
        )

        assert "model_state_dict" in checkpoint
        assert "model_type" in checkpoint
        assert "model_config" in checkpoint
        assert "train_config" in checkpoint
        assert "train_hparams" in checkpoint
        assert "feature_config" in checkpoint
        assert "final_metrics" in checkpoint

    def test_train_without_saving_checkpoint(self, mock_dependencies):
        """Test training without saving checkpoint."""
        with patch("qqe.GNN.training.runners.torch.save"):
            model, test_loss, hist, checkpoint = train(
                model_type="gnn",
                epochs=1,
                save_checkpoint=False,
            )

        assert model is not None

    def test_train_with_custom_batch_size(self, mock_dependencies):
        """Test training with custom batch size."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            batch_size=64,
        )

        assert model is not None

    def test_train_with_custom_learning_rate(self, mock_dependencies):
        """Test training with custom learning rate."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            lr=5e-4,
        )

        assert model is not None

    def test_train_with_progress_disabled(self, mock_dependencies):
        """Test training with progress display disabled."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            show_progress=False,
        )

        assert model is not None

    def test_train_returns_tuple(self, mock_dependencies):
        """Test train returns correct tuple."""
        result = train(model_type="gnn", epochs=1)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_train_checkpoint_includes_metrics(self, mock_dependencies):
        """Test checkpoint includes final metrics."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
        )

        assert "final_metrics" in checkpoint
        assert "test_loss" in checkpoint["final_metrics"]

    @patch("qqe.GNN.training.runners.torch.save")
    def test_train_saves_checkpoint_path(self, mock_save, mock_dependencies):
        """Test checkpoint is saved to expected path."""
        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=1,
            loss_type="mse",
            training_mode="global",
            save_checkpoint=True,
        )

        # Check that torch.save was called
        mock_save.assert_called_once()

    def test_train_with_all_hyperparams(self, mock_dependencies):
        """Test training with all possible hyperparameters."""
        train_hparams = {
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 1e-4,
        }

        model, test_loss, hist, checkpoint = train(
            model_type="gnn",
            epochs=2,
            train_hparams=train_hparams,
        )

        assert model is not None
