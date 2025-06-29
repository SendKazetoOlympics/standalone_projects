from sportsam.sam_handler import SAMHandler, SAMModels
from unittest.mock import MagicMock


def test_sammodels_enum():
    assert SAMModels.SMALL.value[1].endswith("small.pt")
    assert SAMModels.LARGE.value[0].startswith("sam2.1_hiera_l")


def test_samhandler_init(monkeypatch, tmp_path):
    # Patch out model download and predictor creation
    monkeypatch.setattr(
        "sportsam.sam_handler.build_sam2_video_predictor", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "sportsam.sam_handler.urllib.request.urlretrieve", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "sportsam.sam_handler.Path.mkdir",
        lambda self, parents=True, exist_ok=True: None,
    )
    monkeypatch.setattr("sportsam.sam_handler.Path.exists", lambda self: True)
    monkeypatch.setattr("sportsam.sam_handler.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "sportsam.sam_handler.torch.backends.mps.is_available", lambda: False
    )
    handler = SAMHandler(frames_path=tmp_path, model=SAMModels.SMALL)
    assert handler.frames_path == tmp_path
    assert handler.model == SAMModels.SMALL
