import pytest
from unittest.mock import MagicMock, patch

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
        # Patch get_device_properties to return a mock with .major attribute
        dev_prop_mock = MagicMock()
        dev_prop_mock.major = 8
        get_dev_prop_patch.return_value = dev_prop_mock

        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        # The device type used in build_sam2_video_predictor should match expected
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
        # config_path.exists() -> False, checkpoint_path.exists() -> False
        "sportsam.sam_handler.Path.exists",
        side_effect=[False, False],
    ), patch(
        "sportsam.sam_handler.torch.cuda.is_available", return_value=False
    ), patch(
        "sportsam.sam_handler.torch.backends.mps.is_available", return_value=False
    ):
        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        # Should attempt to download config and checkpoint
        assert urlretrieve_mock.call_count == 2


def test_request_prompt_and_not_implemented(tmp_path):
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
        handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
        with pytest.raises(NotImplementedError):
            handler.request_prompt((10, 20))
        with pytest.raises(NotImplementedError):
            handler.init_inference_state()
        with pytest.raises(NotImplementedError):
            handler.analyze_videos("some_dir", MagicMock())
