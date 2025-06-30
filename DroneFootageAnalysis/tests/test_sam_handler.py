import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys

# Patch sys.modules to mock torch and other heavy dependencies
sys.modules["torch"] = MagicMock()
sys.modules["jaxtyping"] = MagicMock()
sys.modules["sam2.build_sam"] = MagicMock()
sys.modules["sam2.sam2_video_predictor"] = MagicMock()
sys.modules["sam2.utils.misc"] = MagicMock()

from sportsam.sam_handler import SAMHandler, SAMModels


@pytest.mark.parametrize(
    "enum_member, expected_yaml, expected_pt",
    [
        (SAMModels.TINY, "sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        (SAMModels.SMALL, "sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        (SAMModels.BASE_PLUS, "sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        (SAMModels.LARGE, "sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    ],
)
def test_sammodels_enum_values(enum_member, expected_yaml, expected_pt):
    assert enum_member.value[0] == expected_yaml
    assert enum_member.value[1] == expected_pt


def setup_handler(
    tmp_path, model=SAMModels.SMALL, config_exists=True, checkpoint_exists=True
):
    # Helper to patch all dependencies for SAMHandler construction
    patches = [
        patch(
            "sportsam.sam_handler.build_sam2_video_predictor", return_value=MagicMock()
        ),
        patch("sportsam.sam_handler.urllib.request.urlretrieve", return_value=None),
        patch("sportsam.sam_handler.Path.mkdir", return_value=None),
        patch("sportsam.sam_handler.torch.cuda.is_available", return_value=False),
        patch(
            "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
        ),
    ]
    # Patch Path.exists for config and checkpoint
    exists_side_effect = [config_exists, checkpoint_exists]
    patches.append(
        patch("sportsam.sam_handler.Path.exists", side_effect=exists_side_effect)
    )
    ctxs = [p for p in patches]
    for ctx in ctxs:
        ctx.start()
    handler = SAMHandler(frames_path=tmp_path, model=model)
    for ctx in ctxs:
        ctx.stop()
    return handler


def test_samhandler_accepts_enum_and_str(tmp_path):
    with patch(
        "sportsam.sam_handler.build_sam2_video_predictor", return_value=MagicMock()
    ), patch(
        "sportsam.sam_handler.urllib.request.urlretrieve", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.mkdir", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.exists", return_value=True
    ), patch(
        "sportsam.sam_handler.torch.cuda.is_available", return_value=False
    ), patch(
        "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
    ):
        # Accepts enum
        handler_enum = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        assert handler_enum.frames_path == tmp_path
        assert handler_enum.model == SAMModels.SMALL

        # Accepts string (enum name)
        handler_str = SAMHandler(frames_path=str(tmp_path), model="SMALL")
        assert handler_str.frames_path == tmp_path
        assert handler_str.model == SAMModels.SMALL


def test_samhandler_invalid_model_raises(tmp_path):
    with patch(
        "sportsam.sam_handler.build_sam2_video_predictor", return_value=MagicMock()
    ), patch(
        "sportsam.sam_handler.urllib.request.urlretrieve", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.mkdir", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.exists", return_value=True
    ), patch(
        "sportsam.sam_handler.torch.cuda.is_available", return_value=False
    ), patch(
        "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
    ):
        with pytest.raises(ValueError):
            SAMHandler(frames_path=tmp_path, model="NOT_A_MODEL")


@pytest.mark.parametrize(
    "cuda, mps, expected_device_type",
    [
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_samhandler_device_selection(tmp_path, cuda, mps, expected_device_type):
    with patch(
        "sportsam.sam_handler.build_sam2_video_predictor", return_value=MagicMock()
    ) as build_mock, patch(
        "sportsam.sam_handler.urllib.request.urlretrieve", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.mkdir", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.exists", return_value=True
    ), patch(
        "sportsam.sam_handler.torch.cuda.is_available", return_value=cuda
    ), patch(
        "sportsam.sam_handler.torch.backends.mps.is_available", return_value=mps
    ), patch(
        "sportsam.sam_handler.torch.device"
    ) as device_patch, patch(
        "sportsam.sam_handler.torch.autocast"
    ), patch(
        "sportsam.sam_handler.torch.cuda.get_device_properties"
    ) as get_dev_prop_patch, patch(
        "sportsam.sam_handler.torch.backends.cuda"
    ), patch(
        "sportsam.sam_handler.torch.backends.cudnn"
    ):
        device_mock = MagicMock()
        device_mock.type = expected_device_type
        device_patch.return_value = device_mock
        dev_prop_mock = MagicMock()
        dev_prop_mock.major = 8
        get_dev_prop_patch.return_value = dev_prop_mock

        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        build_mock.assert_called()
        args, kwargs = build_mock.call_args
        assert kwargs.get("device") == expected_device_type


def test_samhandler_downloads_if_missing(tmp_path):
    # Simulate config and checkpoint not existing
    with patch(
        "sportsam.sam_handler.build_sam2_video_predictor", return_value=MagicMock()
    ), patch(
        "sportsam.sam_handler.urllib.request.urlretrieve", return_value=None
    ) as urlretrieve_mock, patch(
        "sportsam.sam_handler.Path.mkdir", return_value=None
    ), patch(
        "sportsam.sam_handler.Path.exists", side_effect=[False, False]
    ), patch(
        "sportsam.sam_handler.torch.cuda.is_available", return_value=False
    ), patch(
        "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
    ):
        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        # Should attempt to download config and checkpoint
        assert urlretrieve_mock.call_count == 2


def test_analyze_videos_runs(tmp_path):
    # Setup batch directories and mock predictor
    batch0 = tmp_path / "batch0"
    batch0.mkdir(exist_ok=True)
    with patch(
        "sportsam.sam_handler.build_sam2_video_predictor"
    ) as mock_builder, patch("sportsam.sam_handler.urllib.request.urlretrieve"), patch(
        "sportsam.sam_handler.Path.mkdir"
    ), patch(
        "sportsam.sam_handler.Path.exists", return_value=True
    ), patch(
        "sportsam.sam_handler.torch.cuda.is_available", return_value=False
    ), patch(
        "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
    ), patch(
        "sportsam.sam_handler.load_video_frames"
    ) as mock_load_frames:
        mock_predictor = MagicMock()
        mock_predictor.image_size = (224, 224)
        mock_predictor.device = "cpu"
        mock_predictor.init_state.return_value = {"images": [], "num_frames": 0}
        mock_predictor.propagate_in_video.return_value = [(0, [1], "mask_tensor")]
        mock_builder.return_value = mock_predictor
        mock_load_frames.return_value = (["img1", "img2"], None, None)

        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        results = handler.analyze_videos(tmp_path)
        assert isinstance(results, list)
        assert results[0][2] == "mask_tensor"
