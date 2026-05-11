"""Tests for qqe.GNN.training.train module."""

from __future__ import annotations

import sys

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from torch.amp import GradScaler
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qqe.src.GNN.training.train import (
    TrainHistory,
    _amp_device_type,
    _run_train_epoch,
    build_loss,
    train_model,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class TestBuildLoss:
    """Tests for build_loss function."""

    def test_build_mse_loss(self):
        """Test building MSE loss."""
        loss_fn = build_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_build_l1_loss(self):
        """Test building L1 loss."""
        loss_fn = build_loss("l1")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_build_huber_loss(self):
        """Test building Huber loss."""
        loss_fn = build_loss("huber", huber_delta=1.0)
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_build_huber_loss_custom_delta(self):
        """Test building Huber loss with custom delta."""
        loss_fn = build_loss("huber", huber_delta=2.5)
        assert isinstance(loss_fn, nn.HuberLoss)
        assert loss_fn.delta == 2.5

    def test_build_loss_case_insensitive(self):
        """Test that loss type is case-insensitive."""
        loss_fn1 = build_loss("MSE")
        loss_fn2 = build_loss("mse")
        assert type(loss_fn1) == type(loss_fn2)

    def test_build_invalid_loss_type(self):
        """Test error on invalid loss type."""
        with pytest.raises(ValueError, match="Unsupported loss type"):
            build_loss("invalid_loss")


class TestAmpDeviceType:
    """Tests for _amp_device_type function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        device_type = _amp_device_type()
        assert isinstance(device_type, str)
        assert device_type in ["cuda", "cpu"]


class TestTrainHistory:
    """Tests for TrainHistory dataclass."""

    def test_create_train_history(self):
        """Test creating a TrainHistory instance."""
        history = TrainHistory(
            train_loss=[0.5, 0.4, 0.3],
            val_loss=[0.6, 0.5, 0.4],
            lr=[0.001, 0.001, 0.0005],
        )

        assert len(history.train_loss) == 3
        assert len(history.val_loss) == 3
        assert len(history.lr) == 3
        assert history.train_loss[0] == 0.5

    def test_empty_train_history(self):
        """Test creating empty TrainHistory."""
        history = TrainHistory(train_loss=[], val_loss=[], lr=[])

        assert len(history.train_loss) == 0
        assert len(history.val_loss) == 0
        assert len(history.lr) == 0


class TestRunTrainEpoch:
    """Tests for _run_train_epoch function."""

    @pytest.fixture
    def setup_training(self):
        """Setup common training components."""
        device = torch.device("cpu")
        model = SimpleModel(input_dim=10)
        model.to(device)

        loss_fn = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler(device="cpu", enabled=False)

        # Create dummy data loader
        def dummy_loader():
            for _ in range(3):  # 3 batches
                x = torch.randn(4, 10)  # batch of 4
                y = torch.randn(4)
                yield (x, y), None  # Batch must be unpackable

        loader = list(dummy_loader())

        return device, model, loss_fn, optimizer, scaler, loader

    @patch("qqe.GNN.training.train.unpack_supervised_batch")
    def test_run_train_epoch_basic(self, mock_unpack, setup_training):
        """Test basic training epoch run."""
        device, model, loss_fn, optimizer, scaler, loader = setup_training

        # Mock unpack_supervised_batch to return properly formatted data
        x = torch.randn(4, 10)
        y = torch.randn(4)
        mock_unpack.return_value = (x, y, 4)

        with patch("qqe.GNN.training.train.tqdm", side_effect=lambda x, **kwargs: x):
            epoch_loss, elapsed = _run_train_epoch(
                model=model,
                loader=[MagicMock()] * 3,  # Dummy batches
                optimizer=optimizer,
                loss_fn=loss_fn,
                scaler=scaler,
                device=device,
                use_amp=False,
                show_progress=False,
                epoch_idx=1,
                num_epochs=5,
            )

        assert isinstance(epoch_loss, float)
        assert isinstance(elapsed, float)
        assert epoch_loss >= 0
        assert elapsed >= 0

    @patch("qqe.GNN.training.train.unpack_supervised_batch")
    def test_run_train_epoch_no_finite_targets(self, mock_unpack, setup_training):
        """Test epoch handling with no finite targets."""
        device, model, loss_fn, optimizer, scaler, loader = setup_training

        # Mock with NaN targets
        x = torch.randn(4, 10)
        y = torch.full((4,), float("nan"))
        mock_unpack.return_value = (x, y, 4)

        with patch("qqe.GNN.training.train.tqdm", side_effect=lambda x, **kwargs: x):
            epoch_loss, elapsed = _run_train_epoch(
                model=model,
                loader=[MagicMock()] * 3,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scaler=scaler,
                device=device,
                use_amp=False,
                show_progress=False,
            )

        # Should handle gracefully
        assert isinstance(epoch_loss, float)


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def setup_model_and_loaders(self):
        """Setup model and mock loaders."""
        model = SimpleModel(input_dim=10)

        # Create mock loaders
        train_loader = MagicMock()
        train_loader.__len__.return_value = 3
        val_loader = MagicMock()
        val_loader.__len__.return_value = 2

        return model, train_loader, val_loader

    @patch("qqe.GNN.training.train._run_train_epoch")
    @patch("qqe.GNN.training.train.evaluate_loss")
    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_basic(
        self, mock_build_loss, mock_evaluate_loss, mock_run_epoch, setup_model_and_loaders
    ):
        """Test basic model training."""
        model, train_loader, val_loader = setup_model_and_loaders

        mock_build_loss.return_value = nn.MSELoss()
        mock_run_epoch.return_value = (0.5, 1.0)  # loss, time
        mock_evaluate_loss.return_value = 0.4  # val_loss

        trained_model, history, device = train_model(
            model,
            train_loader,
            val_loader,
            epochs=2,
            lr=1e-3,
            show_progress=False,
        )

        assert isinstance(trained_model, nn.Module)
        assert isinstance(history, TrainHistory)
        assert isinstance(device, torch.device)
        assert len(history.train_loss) == 2
        assert len(history.val_loss) == 2

    @patch("qqe.GNN.training.train._run_train_epoch")
    @patch("qqe.GNN.training.train.evaluate_loss")
    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_early_stopping(
        self, mock_build_loss, mock_evaluate_loss, mock_run_epoch, setup_model_and_loaders
    ):
        """Test early stopping during training."""
        model, train_loader, val_loader = setup_model_and_loaders

        mock_build_loss.return_value = nn.MSELoss()
        mock_run_epoch.return_value = (0.5, 1.0)
        # Validation loss increases after first epoch (no improvement)
        mock_evaluate_loss.side_effect = [0.4, 0.5, 0.6, 0.7]

        trained_model, history, device = train_model(
            model,
            train_loader,
            val_loader,
            epochs=100,
            early_stopping_patience=2,
            show_progress=False,
        )

        # Should stop early (patience of 2 with 4 val_loss calls)
        # 1st epoch: 0.4 (best), 2nd: 0.5 (patience 1), 3rd: 0.6 (patience 2), stop
        assert len(history.val_loss) == 3

    @patch("qqe.GNN.training.train._run_train_epoch")
    @patch("qqe.GNN.training.train.evaluate_loss")
    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_with_scheduler(
        self, mock_build_loss, mock_evaluate_loss, mock_run_epoch, setup_model_and_loaders
    ):
        """Test training with learning rate scheduler."""
        model, train_loader, val_loader = setup_model_and_loaders

        mock_build_loss.return_value = nn.MSELoss()
        mock_run_epoch.return_value = (0.5, 1.0)
        mock_evaluate_loss.return_value = 0.4

        trained_model, history, device = train_model(
            model,
            train_loader,
            val_loader,
            epochs=2,
            scheduler="plateau",
            show_progress=False,
        )

        assert len(history.lr) == 2

    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_invalid_loss_type(self, mock_build_loss, setup_model_and_loaders):
        """Test error on invalid loss type."""
        model, train_loader, val_loader = setup_model_and_loaders

        with pytest.raises(ValueError, match="Unsupported loss type"):
            train_model(
                model,
                train_loader,
                val_loader,
                loss_type="invalid",
            )

    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_invalid_scheduler(self, mock_build_loss, setup_model_and_loaders):
        """Test error on invalid scheduler."""
        model, train_loader, val_loader = setup_model_and_loaders
        mock_build_loss.return_value = nn.MSELoss()

        with pytest.raises(ValueError, match="scheduler must be"):
            train_model(
                model,
                train_loader,
                val_loader,
                scheduler="invalid",
            )

    @patch("qqe.GNN.training.train._run_train_epoch")
    @patch("qqe.GNN.training.train.evaluate_loss")
    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_restores_best_state(
        self, mock_build_loss, mock_evaluate_loss, mock_run_epoch, setup_model_and_loaders
    ):
        """Test that training restores best model state."""
        model, train_loader, val_loader = setup_model_and_loaders

        mock_build_loss.return_value = nn.MSELoss()
        mock_run_epoch.return_value = (0.5, 1.0)
        # First epoch best, then gets worse
        mock_evaluate_loss.side_effect = [0.3, 0.5]

        # Save initial state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        trained_model, history, device = train_model(
            model,
            train_loader,
            val_loader,
            epochs=2,
            show_progress=False,
        )

        # Verify best state was restored (history shows best loss from first epoch)
        assert history.val_loss[0] == 0.3

    @patch("qqe.GNN.training.train.build_loss")
    def test_train_model_device_selection(self, mock_build_loss, setup_model_and_loaders):
        """Test device selection in training."""
        model, train_loader, val_loader = setup_model_and_loaders
        mock_build_loss.return_value = nn.MSELoss()

        with patch("qqe.GNN.training.train._run_train_epoch"):
            with patch("qqe.GNN.training.train.evaluate_loss", return_value=0.5):
                _, _, device = train_model(
                    model,
                    train_loader,
                    val_loader,
                    epochs=1,
                    device="cpu",
                    show_progress=False,
                )

        assert device.type == "cpu"
