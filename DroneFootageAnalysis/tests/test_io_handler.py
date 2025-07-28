import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import shutil
import csv

from sportsam.io_handler import IOHandler


@pytest.fixture
def temp_and_output_dirs(tmp_path):
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    temp_dir.mkdir()
    output_dir.mkdir()
    return temp_dir, output_dir


def test_iohandler_init_success(temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    assert handler.temp_dir == temp_dir
    assert handler.output_dir == output_dir


def test_iohandler_init_missing_dirs(tmp_path):
    temp_dir = tmp_path / "missing_temp"
    output_dir = tmp_path / "missing_output"
    with pytest.raises(FileNotFoundError):
        IOHandler(temp_dir, output_dir)
    temp_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        IOHandler(temp_dir, output_dir)


def test_extract_frames_from_videos_empty_list(temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    with pytest.raises(ValueError, match="No video paths provided"):
        handler.extract_frames_from_videos([])


def test_extract_frames_from_videos_file_not_found(temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    with pytest.raises(FileNotFoundError):
        handler.extract_frames_from_videos(["/nonexistent/file.mp4"])


def test_extract_frames_from_videos_unsupported_format(temp_and_output_dirs, tmp_path):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    fake_file = tmp_path / "file.txt"
    fake_file.write_text("not a video")
    with pytest.raises(ValueError):
        handler.extract_frames_from_videos([str(fake_file)])


def test_extract_frames_from_videos_supported_video(monkeypatch, temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    video_file = temp_dir / "video.mp4"
    video_file.write_bytes(b"fake video content")

    class DummyVR:
        def __iter__(self):
            class DummyFrame:
                def asnumpy(self):
                    import numpy as np

                    return np.zeros((10, 10, 3), dtype="uint8")

            return iter([DummyFrame(), DummyFrame()])

    monkeypatch.setattr("decord.VideoReader", lambda path: DummyVR())
    monkeypatch.setattr("imageio.imwrite", lambda path, arr: None)

    handler.extract_frames_from_videos([str(video_file)])


def test_extract_frames_from_videos_jpeg_dir(monkeypatch, temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    img_dir = temp_dir / "frames"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"fake")
    (img_dir / "b.jpeg").write_bytes(b"fake")

    monkeypatch.setattr("shutil.copy", lambda src, dst: shutil.copyfile(src, dst))

    handler.extract_frames_from_videos([str(img_dir)])


def test_extract_frames_from_manifest(monkeypatch, temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    manifest = temp_dir / "manifest.csv"
    video_file = temp_dir / "video.mp4"
    video_file.write_bytes(b"fake")
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([str(video_file)])

    called = {}

    def fake_extract(videos):
        called["videos"] = videos

    monkeypatch.setattr(handler, "extract_frames_from_videos", fake_extract)
    handler.extract_frames_from_manifest(str(manifest))
    assert called["videos"] == [str(video_file)]


def test_extract_frames_from_manifest_file_not_found(temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    with pytest.raises(FileNotFoundError):
        handler.extract_frames_from_manifest("/nonexistent/manifest.csv")


def test_extract_frames_from_manifest_wrong_format(temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    txt_file = temp_dir / "manifest.txt"
    txt_file.write_text("not a csv")
    with pytest.raises(ValueError):
        handler.extract_frames_from_manifest(str(txt_file))

def test_batch_and_unbatch_frames(temp_and_output_dirs):
    temp_dir, output_dir = temp_and_output_dirs
    handler = IOHandler(temp_dir, output_dir)
    
    # Create some test frames
    for i in range(5):
        (temp_dir / f"{i:05d}.jpg").write_bytes(b"fake frame")
    
    # Test batching
    handler.batch_frames(batch_size=2)  # Small batch size for testing
    
    # Check batches were created correctly
    batch_dirs = sorted(d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith("batch"))
    assert len(batch_dirs) == 3  # Should create 3 batches: 2 + 2 + 1 frames
    
    frames_in_batches = []
    for batch_dir in batch_dirs:
        frames = sorted(list(batch_dir.glob("*.jpg")))
        frames_in_batches.extend(frames)
    assert len(frames_in_batches) == 5  # All frames accounted for
    
    # Test unbatching
    handler.unbatch_frames()
    
    # Verify all frames are back in temp_dir
    frames = sorted(list(temp_dir.glob("*.jpg")))
    assert len(frames) == 5
    
    # Verify no batch directories remain
    remaining_batch_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith("batch")]
    assert len(remaining_batch_dirs) == 0
