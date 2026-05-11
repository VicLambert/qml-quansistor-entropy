"""Tests for qqe.GNN.training.datasets module."""

from __future__ import annotations

import sys

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qqe.src.GNN.training.datasets import (
    GlobalTargetDatasetWrapper,
    PaddedGraphDatasetWrapper,
    PreparedData,
    build_loaders,
    build_loaders_NN,
    make_loaders,
    prepare_datasets,
)


@pytest.fixture
def mock_dataset():
    """Create a mock PyG dataset with graph data."""

    class MockPyGDataset(Dataset):
        def __init__(self, num_samples=5):
            self.num_samples = num_samples
            self.all_gate_keys = ["rx_bin_0", "ry_bin_0", "CNOT_count"]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Return a Data object with node features and global features
            data = Data(
                x=torch.randn(3, 8),  # 3 nodes with 8 features each
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                y=torch.tensor(0.5, dtype=torch.float32),
                global_features=torch.randn(3),
                num_graphs=1,
            )
            return data

    return MockPyGDataset()


class TestPaddedGraphDatasetWrapper:
    """Tests for PaddedGraphDatasetWrapper."""

    def test_init_with_explicit_target_dim(self, mock_dataset):
        """Test wrapper initialization with explicit target dimension."""
        wrapper = PaddedGraphDatasetWrapper(mock_dataset, target_dim=10)
        assert wrapper.target_dim == 10
        assert len(wrapper) == len(mock_dataset)

    def test_init_computes_max_dim(self, mock_dataset):
        """Test wrapper computes max dimension when target_dim not provided."""
        wrapper = PaddedGraphDatasetWrapper(mock_dataset)
        assert wrapper.target_dim == 8  # mock_dataset has features of size 8

    def test_getitem_no_padding_needed(self, mock_dataset):
        """Test getting item that doesn't need padding."""
        wrapper = PaddedGraphDatasetWrapper(mock_dataset, target_dim=8)
        data = wrapper[0]
        assert data.x.shape[1] == 8

    def test_getitem_with_padding(self, mock_dataset):
        """Test getting item that needs padding."""
        wrapper = PaddedGraphDatasetWrapper(mock_dataset, target_dim=12)
        data = wrapper[0]
        assert data.x.shape[1] == 12
        # Check that padding was applied with zeros
        assert torch.allclose(data.x[:, 8:], torch.zeros(3, 4))

    def test_len(self, mock_dataset):
        """Test __len__ method."""
        wrapper = PaddedGraphDatasetWrapper(mock_dataset, target_dim=10)
        assert len(wrapper) == len(mock_dataset)


class TestGlobalTargetDatasetWrapper:
    """Tests for GlobalTargetDatasetWrapper."""

    def test_getitem_valid_data(self, mock_dataset):
        """Test getting item with valid data."""
        wrapper = GlobalTargetDatasetWrapper(mock_dataset)
        g, y = wrapper[0]

        assert isinstance(g, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert g.dim() == 1
        assert y.dim() == 0  # scalar

    def test_getitem_missing_global_features(self):
        """Test error when global_features are missing."""

        class BadDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return Data(x=torch.randn(2, 3), y=torch.tensor(0.5))

        wrapper = GlobalTargetDatasetWrapper(BadDataset())
        with pytest.raises(ValueError, match="missing 'global_features'"):
            wrapper[0]

    def test_getitem_nan_target(self):
        """Test handling of NaN target values."""

        class DatasetWithNaN(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return Data(
                    x=torch.randn(2, 3),
                    y=None,
                    global_features=torch.randn(3),
                )

        wrapper = GlobalTargetDatasetWrapper(DatasetWithNaN())
        g, y = wrapper[0]
        assert torch.isnan(y)

    def test_len(self, mock_dataset):
        """Test __len__ method."""
        wrapper = GlobalTargetDatasetWrapper(mock_dataset)
        assert len(wrapper) == len(mock_dataset)


class TestPreparedData:
    """Tests for PreparedData dataclass."""

    def test_dataclass_creation(self):
        """Test creating a PreparedData instance."""
        prepared = PreparedData(
            train_ds=MagicMock(),
            val_ds=MagicMock(),
            test_ds=MagicMock(),
            node_in_dim=8,
            global_in_dim=64,
            base_dataset=MagicMock(),
            loader_kind="gnn",
        )

        assert prepared.node_in_dim == 8
        assert prepared.global_in_dim == 64
        assert prepared.loader_kind == "gnn"

    def test_dataclass_with_none_node_dim(self):
        """Test PreparedData with None node_in_dim."""
        prepared = PreparedData(
            train_ds=MagicMock(),
            val_ds=MagicMock(),
            test_ds=MagicMock(),
            node_in_dim=None,
            global_in_dim=64,
            base_dataset=MagicMock(),
            loader_kind="nn",
        )

        assert prepared.node_in_dim is None


@pytest.mark.slow
class TestPrepareDatasets:
    """Tests for prepare_datasets function."""

    @patch("qqe.GNN.training.datasets.QuantumCircuitGraphDataset")
    def test_prepare_datasets_gnn_loader(self, mock_dataset_class):
        """Test prepare_datasets with GNN loader."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.__len__.return_value = 100
        mock_instance.all_gate_keys = ["gate1", "gate2"]

        sample_data = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor(0.5),
            global_features=torch.randn(3),
            num_graphs=1,
        )
        mock_instance.__getitem__.return_value = sample_data
        mock_dataset_class.return_value = mock_instance

        with patch("qqe.GNN.training.datasets.cache_root_paths", return_value="cache_dir"):
            with patch("qqe.GNN.training.datasets.PaddedGraphDatasetWrapper"):
                prepared = prepare_datasets(
                    ["dummy.pt"],
                    loader_kind="gnn",
                    seed=42,
                )

        assert prepared.loader_kind == "gnn"
        assert prepared.node_in_dim == 8
        assert prepared.global_in_dim == 3

    @patch("qqe.GNN.training.datasets.QuantumCircuitGraphDataset")
    def test_prepare_datasets_nn_loader(self, mock_dataset_class):
        """Test prepare_datasets with NN loader."""
        mock_instance = MagicMock()
        mock_instance.__len__.return_value = 100
        mock_instance.all_gate_keys = ["gate1", "gate2"]

        sample_data = Data(
            x=torch.randn(5, 8),
            y=torch.tensor(0.5),
            global_features=torch.randn(3),
        )
        mock_instance.__getitem__.return_value = sample_data
        mock_dataset_class.return_value = mock_instance

        with patch("qqe.GNN.training.datasets.cache_root_paths", return_value="cache_dir"):
            with patch("qqe.GNN.training.datasets.GlobalTargetDatasetWrapper"):
                prepared = prepare_datasets(
                    ["dummy.pt"],
                    loader_kind="nn",
                    seed=42,
                )

        assert prepared.loader_kind == "nn"
        assert prepared.node_in_dim is None
        assert prepared.global_in_dim == 3

    @patch("qqe.GNN.training.datasets.QuantumCircuitGraphDataset")
    def test_prepare_datasets_invalid_loader_kind(self, mock_dataset_class):
        """Test prepare_datasets with invalid loader_kind."""
        mock_instance = MagicMock()
        mock_instance.__len__.return_value = 100
        mock_dataset_class.return_value = mock_instance

        with patch("qqe.GNN.training.datasets.cache_root_paths", return_value="cache_dir"):
            with pytest.raises(ValueError, match="loader_kind must be"):
                prepare_datasets(
                    ["dummy.pt"],
                    loader_kind="invalid",
                )

    @patch("qqe.GNN.training.datasets.QuantumCircuitGraphDataset")
    def test_prepare_datasets_too_small(self, mock_dataset_class):
        """Test prepare_datasets with dataset that's too small."""
        mock_instance = MagicMock()
        mock_instance.__len__.return_value = 2  # Too small
        mock_dataset_class.return_value = mock_instance

        with patch("qqe.GNN.training.datasets.cache_root_paths", return_value="cache_dir"):
            with pytest.raises(RuntimeError, match="Dataset too small"):
                prepare_datasets(["dummy.pt"], loader_kind="gnn")


@pytest.mark.slow
class TestMakeLoaders:
    """Tests for make_loaders function."""

    def test_make_loaders_gnn(self):
        """Test make_loaders with GNN prepared data."""
        # Create mock datasets
        train_ds = MagicMock()
        train_ds.__len__.return_value = 50

        prepared = PreparedData(
            train_ds=train_ds,
            val_ds=MagicMock(),
            test_ds=MagicMock(),
            node_in_dim=8,
            global_in_dim=64,
            base_dataset=MagicMock(),
            loader_kind="gnn",
        )

        with patch("qqe.GNN.training.datasets.PyGDataLoader"):
            loaders = make_loaders(prepared, batch_size=32, num_workers=0)

            assert len(loaders) == 3
            # All should be PyGDataLoader instances (mocked)

    def test_make_loaders_nn(self):
        """Test make_loaders with NN prepared data."""
        prepared = PreparedData(
            train_ds=MagicMock(),
            val_ds=MagicMock(),
            test_ds=MagicMock(),
            node_in_dim=None,
            global_in_dim=64,
            base_dataset=MagicMock(),
            loader_kind="nn",
        )

        with patch("qqe.GNN.training.datasets.TorchDataLoader"):
            loaders = make_loaders(prepared, batch_size=32, num_workers=0)

            assert len(loaders) == 3

    def test_make_loaders_invalid_kind(self):
        """Test make_loaders with invalid loader_kind."""
        prepared = PreparedData(
            train_ds=MagicMock(),
            val_ds=MagicMock(),
            test_ds=MagicMock(),
            node_in_dim=8,
            global_in_dim=64,
            base_dataset=MagicMock(),
            loader_kind="invalid",
        )

        with pytest.raises(ValueError, match="Unknown loader_kind"):
            make_loaders(prepared, batch_size=32)


@pytest.mark.slow
@patch("qqe.GNN.training.datasets.prepare_datasets")
@patch("qqe.GNN.training.datasets.make_loaders")
class TestBuildLoaders:
    """Tests for build_loaders and build_loaders_NN functions."""

    def test_build_loaders(self, mock_make_loaders, mock_prepare):
        """Test build_loaders function."""
        # Setup mocks
        mock_prepared = MagicMock()
        mock_prepared.loader_kind = "gnn"
        mock_prepare.return_value = mock_prepared

        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_make_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)

        result = build_loaders(
            ["path/to/data.pt"],
            batch_size=32,
            seed=42,
        )

        assert len(result) == 6
        assert result[0] == mock_train_loader
        assert result[1] == mock_val_loader
        assert result[2] == mock_test_loader

    def test_build_loaders_nn(self, mock_make_loaders, mock_prepare):
        """Test build_loaders_NN function."""
        # Setup mocks
        mock_prepared = MagicMock()
        mock_prepared.loader_kind = "nn"
        mock_prepare.return_value = mock_prepared

        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_make_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)

        result = build_loaders_NN(
            ["path/to/data.pt"],
            batch_size=32,
            seed=42,
        )

        assert len(result) == 5  # NN version returns 5 items (no node_in_dim)
        assert result[0] == mock_train_loader
        assert result[1] == mock_val_loader
        assert result[2] == mock_test_loader
