"""Tests for qqe.GNN.training.utils module."""

from __future__ import annotations

import sys
import tempfile

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qqe.GNN.training.train_config import FAMILY_GATE_TYPES, MASTER_GATE_TYPES
from qqe.GNN.training.utils import (
    FamilyFeatureProjector,
    FamilyGlobalProjector,
    FamilyNodeProjector,
    ProjectedDatasetWrapper,
    _amp_device_type,
    _family_global_gate_keys,
    _move_to_device,
    _safe_y,
    cache_root_paths,
    collect_files_path,
    evaluate_loss,
    unpack_supervised_batch,
)


class TestAmpDeviceType:
    """Tests for _amp_device_type function."""

    def test_returns_valid_device_type(self):
        """Test that function returns valid device type."""
        device_type = _amp_device_type()
        assert device_type in ["cuda", "cpu"]

    def test_returns_string(self):
        """Test return type is string."""
        device_type = _amp_device_type()
        assert isinstance(device_type, str)


class TestFamilyGlobalGateKeys:
    """Tests for _family_global_gate_keys function."""

    def test_random_family_keys(self):
        """Test gate keys selection for random family."""
        all_keys = ["rx_bin_0", "rx_bin_1", "ry_bin_0", "rz_bin_0", "CNOT_count", "other"]
        keys = _family_global_gate_keys("random", all_keys)

        assert "rx_bin_0" in keys
        assert "ry_bin_0" in keys
        assert "CNOT_count" in keys
        assert "other" not in keys

    def test_clifford_family_keys(self):
        """Test gate keys selection for clifford family."""
        all_keys = ["I_count", "H_count", "S_count", "T_count", "CNOT_count", "other"]
        keys = _family_global_gate_keys("clifford", all_keys)

        assert "I_count" in keys
        assert "H_count" in keys
        assert "S_count" in keys
        assert "T_count" in keys
        assert "CNOT_count" in keys
        assert "other" not in keys

    def test_haar_family_keys(self):
        """Test gate keys selection for haar family."""
        all_keys = ["haar_eig_bin_0", "haar_eig_bin_1", "other"]
        keys = _family_global_gate_keys("haar", all_keys)

        assert "haar_eig_bin_0" in keys
        assert "haar_eig_bin_1" in keys
        assert "other" not in keys

    def test_quansistor_family_keys(self):
        """Test gate keys selection for quansistor family."""
        all_keys = ["qx_bin_0", "qy_bin_0", "qx_bin_1", "other"]
        keys = _family_global_gate_keys("quansistor", all_keys)

        assert "qx_bin_0" in keys
        assert "qy_bin_0" in keys
        assert "qx_bin_1" in keys
        assert "other" not in keys

    def test_invalid_family(self):
        """Test error on invalid family."""
        with pytest.raises(ValueError, match="Unknown family"):
            _family_global_gate_keys("invalid", [])


class TestFamilyNodeProjector:
    """Tests for FamilyNodeProjector class."""

    def test_init_random_family(self):
        """Test initializing projector for random family."""
        projector = FamilyNodeProjector("random")
        assert projector.family == "random"
        assert len(projector.keep_gate_idx) > 0

    def test_call_with_valid_data(self):
        """Test projector call with valid data."""
        projector = FamilyNodeProjector("random")

        # Create test data: features should have shape (num_nodes, num_features)
        # where num_features >= len(MASTER_GATE_TYPES) + qubit_mask_size
        num_master = len(MASTER_GATE_TYPES)
        num_qubits = 4
        x = torch.randn(10, num_master + num_qubits)  # 10 nodes

        data = Data(
            x=x,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor(0.5),
        )

        result = projector(data)

        assert isinstance(result, Data)
        assert result.x.shape[0] == 10  # Same number of nodes
        # Shape should be (num_nodes, len(keep_gate_idx) + qubit_mask_size)
        assert result.x.shape[1] <= x.shape[1]

    def test_different_families(self):
        """Test projector with different families."""
        for family in FAMILY_GATE_TYPES.keys():
            projector = FamilyNodeProjector(family)
            assert projector.family == family


class TestFamilyGlobalProjector:
    """Tests for FamilyGlobalProjector class."""

    def test_init(self):
        """Test initializing global projector."""
        all_gate_keys = ["I_count", "H_count", "S_count", "T_count", "CNOT_count"]
        projector = FamilyGlobalProjector("clifford", all_gate_keys)

        assert projector.family == "clifford"
        assert projector.all_gate_keys == all_gate_keys
        assert len(projector.keep_idx) > 0

    def test_call_with_1d_features(self):
        """Test projector with 1D global features."""
        all_gate_keys = ["I_count", "H_count", "S_count"]
        projector = FamilyGlobalProjector("clifford", all_gate_keys)

        # Create data with 1D features: [n_qubits, n_bins] + gate_counts
        global_features = torch.tensor([4.0, 10.0, 1.0, 2.0, 3.0])  # 5 features

        data = Data(
            x=torch.randn(3, 4),
            y=torch.tensor(0.5),
            global_features=global_features,
        )

        result = projector(data)

        assert result.global_features.shape[0] > 0
        assert result.global_features.shape[0] <= global_features.shape[0]

    def test_call_with_2d_features(self):
        """Test projector with 2D global features."""
        all_gate_keys = ["I_count", "H_count"]
        projector = FamilyGlobalProjector("clifford", all_gate_keys)

        # Create data with 2D features (batch)
        global_features = torch.randn(2, 4)  # batch of 2

        data = Data(
            x=torch.randn(3, 4),
            y=torch.tensor(0.5),
            global_features=global_features,
        )

        result = projector(data)

        assert result.global_features.dim() == 2
        assert result.global_features.shape[0] == 2


class TestFamilyFeatureProjector:
    """Tests for FamilyFeatureProjector class."""

    def test_init_creates_sub_projectors(self):
        """Test that FamilyFeatureProjector creates sub-projectors."""
        all_gate_keys = ["I_count", "H_count"]
        projector = FamilyFeatureProjector("clifford", all_gate_keys)

        assert hasattr(projector, "node_projector")
        assert hasattr(projector, "global_projector")

    def test_call_applies_both_projections(self):
        """Test that __call__ applies both node and global projections."""
        all_gate_keys = ["I_count", "H_count"]
        projector = FamilyFeatureProjector("clifford", all_gate_keys)

        num_master = len(MASTER_GATE_TYPES)
        num_qubits = 4
        x = torch.randn(5, num_master + num_qubits)

        data = Data(
            x=x,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor(0.5),
            global_features=torch.randn(4),
        )

        result = projector(data)

        assert isinstance(result, Data)
        assert result.x.shape[0] == 5
        assert result.global_features is not None


class TestProjectedDatasetWrapper:
    """Tests for ProjectedDatasetWrapper class."""

    def test_init(self):
        """Test wrapper initialization."""
        mock_dataset = MagicMock()
        mock_transform = MagicMock()

        wrapper = ProjectedDatasetWrapper(mock_dataset, mock_transform)

        assert wrapper.dataset == mock_dataset
        assert wrapper.transform == mock_transform

    def test_len(self):
        """Test __len__ method."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_transform = MagicMock()

        wrapper = ProjectedDatasetWrapper(mock_dataset, mock_transform)

        assert len(wrapper) == 10

    def test_getitem_applies_transform(self):
        """Test __getitem__ applies transform."""
        mock_dataset = MagicMock()
        mock_data = Data(x=torch.randn(2, 4))
        mock_dataset.__getitem__.return_value = mock_data

        mock_transform = MagicMock(return_value=mock_data)

        wrapper = ProjectedDatasetWrapper(mock_dataset, mock_transform)

        result = wrapper[0]

        mock_transform.assert_called_once_with(mock_data)
        assert result == mock_data


class TestSafeY:
    """Tests for _safe_y function."""

    def test_safe_y_with_scalar(self):
        """Test _safe_y with scalar tensor."""
        y = torch.tensor(0.5)
        result = _safe_y(y)

        assert result.shape == torch.Size([1])
        assert result[0].item() == 0.5

    def test_safe_y_with_1d(self):
        """Test _safe_y with 1D tensor."""
        y = torch.tensor([0.5, 0.3, 0.7])
        result = _safe_y(y)

        assert result.shape == torch.Size([3])

    def test_safe_y_with_2d_single_column(self):
        """Test _safe_y with 2D tensor of shape (n, 1)."""
        y = torch.tensor([[0.5], [0.3], [0.7]])
        result = _safe_y(y)

        assert result.shape == torch.Size([3])

    def test_safe_y_dtype_conversion(self):
        """Test _safe_y converts to float32."""
        y = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = _safe_y(y)

        assert result.dtype == torch.float32


class TestMoveToDevice:
    """Tests for _move_to_device function."""

    def test_move_tensor_to_cpu(self):
        """Test moving tensor to CPU."""
        x = torch.randn(3, 4)
        result = _move_to_device(x, torch.device("cpu"))

        assert result.device.type == "cpu"

    def test_move_data_object_to_device(self):
        """Test moving Data object to device."""
        data = Data(x=torch.randn(2, 3))
        result = _move_to_device(data, torch.device("cpu"))

        assert result.x.device.type == "cpu"

    def test_move_non_tensor_passthrough(self):
        """Test non-tensor objects pass through."""
        obj = {"key": "value"}
        result = _move_to_device(obj, torch.device("cpu"))

        assert result == obj


class TestUnpackSupervisedBatch:
    """Tests for unpack_supervised_batch function."""

    def test_unpack_pyg_batch(self):
        """Test unpacking PyG batch."""
        device = torch.device("cpu")
        batch = Data(
            x=torch.randn(4, 8),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([0.5, 0.3, 0.2, 0.1]),
            num_graphs=4,
        )

        model_input, y, batch_size = unpack_supervised_batch(batch, device)

        assert isinstance(y, torch.Tensor)
        assert y.shape[0] == 4
        assert batch_size == 4

    def test_unpack_tuple_batch(self):
        """Test unpacking tuple batch (x, y)."""
        device = torch.device("cpu")
        x = torch.randn(4, 8)
        y = torch.tensor([0.5, 0.3, 0.2, 0.1])
        batch = (x, y)

        model_input, y_result, batch_size = unpack_supervised_batch(batch, device)

        assert torch.equal(model_input, x)
        assert torch.equal(y_result, y)
        assert batch_size == 4

    def test_unpack_invalid_batch(self):
        """Test error on invalid batch."""
        device = torch.device("cpu")
        batch = (torch.randn(4, 8),)  # Only x, no y

        with pytest.raises(ValueError, match="must contain at least"):
            unpack_supervised_batch(batch, device)


class TestCollectFilesPath:
    """Tests for collect_files_path function."""

    def test_collect_files_from_directory(self):
        """Test collecting files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            family_dir = Path(tmpdir) / "random"
            family_dir.mkdir()

            (family_dir / "data1.pt").touch()
            (family_dir / "data2.pt").touch()

            paths = collect_files_path(tmpdir, family="random")

            assert len(paths) >= 2

    def test_collect_files_all_families(self):
        """Test collecting files from all families."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for family in ["random", "clifford"]:
                family_dir = Path(tmpdir) / family
                family_dir.mkdir()
                (family_dir / f"{family}_data.pt").touch()

            paths = collect_files_path(tmpdir)

            assert len(paths) >= 2

    def test_collect_files_empty_directory(self):
        """Test collecting from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = collect_files_path(tmpdir)

            assert isinstance(paths, list)


class TestCacheRootPaths:
    """Tests for cache_root_paths function."""

    def test_cache_path_generation(self):
        """Test cache path generation."""
        paths = ["data1.pt", "data2.pt"]
        cache_dir = cache_root_paths(paths)

        assert isinstance(cache_dir, str)
        assert "pyg_cache" in cache_dir

    def test_cache_path_consistency(self):
        """Test cache path is consistent for same inputs."""
        paths = ["data1.pt", "data2.pt"]
        cache_dir1 = cache_root_paths(paths)
        cache_dir2 = cache_root_paths(paths)

        assert cache_dir1 == cache_dir2

    def test_cache_path_with_suffix(self):
        """Test cache path includes suffix."""
        paths = ["data1.pt"]
        cache_dir = cache_root_paths(paths, suffix="test_suffix")

        assert "test_suffix" in cache_dir


class TestEvaluateLoss:
    """Tests for evaluate_loss function."""

    def test_evaluate_loss_mse(self):
        """Test loss evaluation with MSE loss."""
        device = torch.device("cpu")
        model = nn.Linear(10, 1)
        loss_fn = nn.MSELoss()

        # Mock loader
        loader = MagicMock()

        def mock_batch():
            x = torch.randn(4, 10)
            y = torch.randn(4)
            return (x, y), None

        loader.__iter__.return_value = [mock_batch()]

        with patch("qqe.GNN.training.utils.unpack_supervised_batch") as mock_unpack:
            mock_unpack.return_value = (torch.randn(4, 10), torch.randn(4), 4)

            loss = evaluate_loss(
                model,
                loader,
                device,
                loss_fn,
                use_amp=False,
                show_progress=False,
            )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_loss_with_nan_targets(self):
        """Test loss evaluation handles NaN targets."""
        device = torch.device("cpu")
        model = nn.Linear(10, 1)
        loss_fn = nn.MSELoss()
        loader = MagicMock()
        loader.__iter__.return_value = []

        # Should handle empty batch gracefully
        loss = evaluate_loss(
            model,
            loader,
            device,
            loss_fn,
            use_amp=False,
        )

        assert isinstance(loss, float)

    def test_evaluate_loss_with_progress(self):
        """Test loss evaluation shows progress."""
        device = torch.device("cpu")
        model = nn.Linear(10, 1)
        loss_fn = nn.MSELoss()
        loader = MagicMock()
        loader.__iter__.return_value = []

        with patch("qqe.GNN.training.utils.tqdm"):
            loss = evaluate_loss(
                model,
                loader,
                device,
                loss_fn,
                use_amp=False,
                show_progress=True,
            )

        assert isinstance(loss, float)
