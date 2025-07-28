"""
Tests for sam_handler.py

Tests the SAMHandler class and its methods.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from sportsam.sam_handler import SAMHandler, SAMModels


class TestSAMModels:
    """Test the SAMModels enum."""

    @pytest.mark.parametrize(
        "model,expected_config,expected_checkpoint",
        [
            (SAMModels.TINY, "sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
            (SAMModels.SMALL, "sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
            (SAMModels.BASE_PLUS, "sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
            (SAMModels.LARGE, "sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        ],
    )
    def test_sam_models_enum_values(self, model, expected_config, expected_checkpoint):
        """Test that SAMModels enum has correct values."""
        assert model.value[0] == expected_config
        assert model.value[1] == expected_checkpoint


class TestSAMHandler:
    """Test the SAMHandler class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for SAMHandler."""
        with patch(
            "sportsam.sam_handler.build_sam2_video_predictor"
        ) as mock_build, patch(
            "sportsam.sam_handler.urllib.request.urlretrieve"
        ) as mock_download, patch(
            "sportsam.sam_handler.torch.cuda.is_available", return_value=False
        ), patch(
            "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
        ), patch(
            "sportsam.sam_handler.Path.mkdir"
        ), patch(
            "sportsam.sam_handler.Path.exists", return_value=True
        ):

            mock_predictor = MagicMock()
            mock_predictor.image_size = (224, 224)
            mock_predictor.device = "cpu"
            mock_build.return_value = mock_predictor

            yield {
                "mock_build": mock_build,
                "mock_download": mock_download,
                "mock_predictor": mock_predictor,
            }

    def test_initialization_with_enum(self, tmp_path):
        """Test SAMHandler initialization with SAMModels enum."""
        with patch('sportsam.sam_handler.Path.exists', return_value=True):
            SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)

    def test_initialization_with_string_model_name(self, tmp_path):
        """Test SAMHandler initialization with string model name."""
        with patch('sportsam.sam_handler.Path.exists', return_value=True):
            SAMHandler(frames_path=str(tmp_path), model="SMALL")

    def test_initialization_with_invalid_model_string(self, tmp_path):
        """Test SAMHandler initialization with invalid model string raises ValueError."""
        with patch('sportsam.sam_handler.Path.exists', return_value=True):
            with pytest.raises(
                ValueError, match="INVALID_MODEL is not a valid SAMModels name or value"
            ):
                SAMHandler(frames_path=tmp_path, model="INVALID_MODEL")

    def test_checkpoint_download_when_missing(self, tmp_path):
        """Test that checkpoint is downloaded when missing."""
        with patch(
            "sportsam.sam_handler.build_sam2_video_predictor"
        ) as mock_build, patch(
            "sportsam.sam_handler.urllib.request.urlretrieve"
        ) as mock_download, patch(
            "sportsam.sam_handler.torch.cuda.is_available", return_value=False
        ), patch(
            "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
        ), patch(
            "sportsam.sam_handler.Path.mkdir"
        ), patch(
            "sportsam.sam_handler.Path.exists", return_value=False
        ):

            mock_predictor = MagicMock()
            mock_build.return_value = mock_predictor

            handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)

            # Verify download was called
            mock_download.assert_called_once()
            args = mock_download.call_args[0]
            assert "sam2.1_hiera_small.pt" in args[0]  # URL contains checkpoint name

    @pytest.mark.parametrize(
        "cuda_available,mps_available,expected_device",
        [
            (True, False, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ],
    )
    def test_device_selection(
        self, cuda_available, mps_available, expected_device, tmp_path
    ):
        """Test that correct device is selected based on availability."""
        with patch(
            "sportsam.sam_handler.build_sam2_video_predictor"
        ) as mock_build, patch(
            "sportsam.sam_handler.urllib.request.urlretrieve"
        ), patch(
            "sportsam.sam_handler.torch.cuda.is_available", return_value=cuda_available
        ), patch(
            "sportsam.sam_handler.torch.backends.mps.is_available",
            return_value=mps_available,
        ), patch(
            "sportsam.sam_handler.Path.mkdir"
        ), patch(
            "sportsam.sam_handler.Path.exists", return_value=True
        ), patch(
            "sportsam.sam_handler.torch.device"
        ) as mock_device, patch(
            "sportsam.sam_handler.torch.autocast"
        ), patch(
            "sportsam.sam_handler.torch.cuda.get_device_properties",
            return_value=MagicMock(major=8, minor=0)
        ), patch(
            "sportsam.sam_handler.torch.backends.cuda"
        ), patch(
            "sportsam.sam_handler.torch.backends.cudnn"
        ):

            device_mock = MagicMock()
            device_mock.type = expected_device
            mock_device.return_value = device_mock

            mock_predictor = MagicMock()
            mock_build.return_value = mock_predictor

            handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)

            # Verify build_sam2_video_predictor was called with correct device
            mock_build.assert_called_once()
            args, kwargs = mock_build.call_args
            assert kwargs["device"] == expected_device

    def test_request_click_file_not_found(self, mock_dependencies, tmp_path):
        """Test request_click raises FileNotFoundError when frame doesn't exist."""
        with patch("sportsam.sam_handler.cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)

        with pytest.raises(FileNotFoundError, match="Image at .* does not exist"):
            handler.request_click(frame_idx=0, obj_id=1)

    def test_request_click_with_valid_frame(self, mock_dependencies, tmp_path):
        """Test request_click with a valid frame file."""
        # Create mock frame file
        batch_dir = tmp_path / "batch0"
        batch_dir.mkdir()
        frame_file = batch_dir / "00000.jpg"
        frame_file.parent.mkdir(parents=True, exist_ok=True)
        frame_file.write_bytes(b"fake image data")

        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        handler.inference_state = {}

        with patch("sportsam.sam_handler.cv2.imread") as mock_imread, patch(
            "sportsam.sam_handler.cv2.namedWindow"
        ), patch("sportsam.sam_handler.cv2.setMouseCallback"), patch(
            "sportsam.sam_handler.np.array"
        ) as mock_array:

            # Mock cv2.imread to return valid image
            mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_array.side_effect = [
                np.array([[(421, 361)]], dtype=np.float32),  # points
                np.array([1], dtype=np.int32),  # labels
            ]

            # Mock predictor method
            handler.predictor.add_new_points_or_box = MagicMock()

            # Call request_click
            handler.request_click(frame_idx=0, obj_id=1)

            # Verify predictor method was called
            handler.predictor.add_new_points_or_box.assert_called_once()

    def test_segment_videos(self, tmp_path):
        """Test segment_videos method."""
        # Create mock batch directories
        batch0 = tmp_path / "batch0"
        batch1 = tmp_path / "batch1"
        batch0.mkdir()
        batch1.mkdir()

        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        handler.inference_state = {}

        with patch("sportsam.sam_handler.load_video_frames") as mock_load_frames:
            # Mock load_video_frames
            mock_images = [torch.zeros((3, 224, 224)) for _ in range(2)]
            mock_load_frames.return_value = (mock_images, None, None)

            # Mock predictor.propagate_in_video
            mock_mask_logits = [torch.ones((1, 100, 100)) * 0.6]  # Above threshold
            mock_propagate_results = [
                (0, [1], mock_mask_logits),
                (1, [1], mock_mask_logits),
            ]
            handler.predictor.propagate_in_video = MagicMock(
                return_value=iter(mock_propagate_results)
            )

            # Call segment_videos
            results = handler.segment_videos(tmp_path)

            # Verify results structure
            assert isinstance(results, dict)
            assert len(results) == 4  # 2 frames per batch, 2 batches

            # Check that all frame results have correct structure
            for frame_idx, frame_result in results.items():
                assert isinstance(frame_result, dict)
                assert 1 in frame_result  # obj_id 1
                assert isinstance(frame_result[1], np.ndarray)  # mask is numpy array

    def test_segment_videos_no_batch_dirs(self, mock_dependencies, tmp_path):
        """Test segment_videos with no batch directories."""
        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)

        results = handler.segment_videos(tmp_path)

        # Should return empty dict when no batch directories found
        assert results == {}

    def test_segment_videos_empty_batch_dir(self, mock_dependencies, tmp_path):
        """Test segment_videos with empty batch directory."""
        # Create empty batch directory
        batch0 = tmp_path / "batch0"
        batch0.mkdir()

        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        handler.inference_state = {}

        with patch("sportsam.sam_handler.load_video_frames") as mock_load_frames:
            # Mock load_video_frames to return empty list
            mock_load_frames.return_value = ([], None, None)

            # Mock predictor.propagate_in_video to return empty iterator
            handler.predictor.propagate_in_video = MagicMock(return_value=iter([]))

            results = handler.segment_videos(tmp_path)

            # Should return empty dict for empty batch
            assert results == {}


class TestSAMHandlerIntegration:
    """Integration tests for SAMHandler (testing with minimal mocking)."""

    def test_initialization_creates_checkpoint_directory(self, tmp_path):
        """Test that initialization creates checkpoint directory."""
        with patch(
            "sportsam.sam_handler.build_sam2_video_predictor"
        ) as mock_build, patch(
            "sportsam.sam_handler.urllib.request.urlretrieve"
        ), patch(
            "sportsam.sam_handler.torch.cuda.is_available", return_value=False
        ), patch(
            "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
        ), patch(
            "sportsam.sam_handler.Path.exists", return_value=True
        ), patch(
            "sportsam.sam_handler.Path.mkdir"
        ) as mock_mkdir:

            mock_predictor = MagicMock()
            mock_build.return_value = mock_predictor

            handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)

            # Verify mkdir was called
            mock_mkdir.assert_called()
            args = mock_mkdir.call_args
            assert args[1]["parents"] is True
            assert args[1]["exist_ok"] is True
