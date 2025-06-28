import shutil
import csv
import pytest

from sportsam.io_handler import IOHandler


@pytest.fixture
def io_handler():
    return IOHandler()


def test_extract_frames_from_videos_empty_list(io_handler):
    with pytest.raises(ValueError, match="No video paths provided"):
        io_handler.extract_frames_from_videos([])


def test_extract_frames_from_videos_file_not_found(io_handler):
    with pytest.raises(FileNotFoundError):
        io_handler.extract_frames_from_videos(["/nonexistent/file.mp4"])


def test_extract_frames_from_videos_unsupported_format(io_handler, tmp_path):
    fake_file = tmp_path / "file.txt"
    fake_file.write_text("not a video")
    with pytest.raises(ValueError):
        io_handler.extract_frames_from_videos([str(fake_file)])


def test_extract_frames_from_videos_supported_video(monkeypatch, io_handler, tmp_path):
    # Create a fake .mp4 file
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake video content")

    # Patch decord.VideoReader and imageio.imwrite
    class DummyVR:
        def __iter__(self):
            class DummyFrame:
                def asnumpy(self):
                    import numpy as np

                    return np.zeros((10, 10, 3), dtype="uint8")

            return iter([DummyFrame(), DummyFrame()])

    monkeypatch.setattr("decord.VideoReader", lambda path: DummyVR())
    monkeypatch.setattr("imageio.imwrite", lambda path, arr: None)

    temp_dir = io_handler.extract_frames_from_videos([str(video_file)])
    assert temp_dir.exists()
    shutil.rmtree(temp_dir)


def test_extract_frames_from_videos_jpeg_dir(monkeypatch, io_handler, tmp_path):
    # Create a directory with JPEGs
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"fake")
    (img_dir / "b.jpeg").write_bytes(b"fake")

    # Patch shutil.copy
    monkeypatch.setattr("shutil.copy", lambda src, dst: shutil.copyfile(src, dst))

    temp_dir = io_handler.extract_frames_from_videos([str(img_dir)])
    assert temp_dir.exists()
    shutil.rmtree(temp_dir)


def test_extract_frames_from_manifest(tmp_path, monkeypatch, io_handler):
    # Create a manifest CSV
    manifest = tmp_path / "manifest.csv"
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake")
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([str(video_file)])

    # Patch extract_frames_from_videos
    monkeypatch.setattr(io_handler, "extract_frames_from_videos", lambda vids: "called")
    result = io_handler.extract_frames_from_manifest(str(manifest))
    assert result == "called"


def test_extract_frames_from_manifest_file_not_found(io_handler):
    with pytest.raises(FileNotFoundError):
        io_handler.extract_frames_from_manifest("/nonexistent/manifest.csv")


def test_extract_frames_from_manifest_wrong_format(tmp_path, io_handler):
    txt_file = tmp_path / "manifest.txt"
    txt_file.write_text("not a csv")
    with pytest.raises(ValueError):
        io_handler.extract_frames_from_manifest(str(txt_file))


def test_clear_temp_dir(tmp_path, io_handler):
    tmp_dir = tmp_path / "todelete"
    tmp_dir.mkdir()
    (tmp_dir / "file.txt").write_text("data")
    io_handler.clear_temp_dir(tmp_dir)
    assert not tmp_dir.exists()


def test_clear_temp_dir_nonexistent(io_handler, tmp_path, capsys):
    non_dir = tmp_path / "notadir"
    io_handler.clear_temp_dir(non_dir)
    captured = capsys.readouterr()
    assert "does not exist" in captured.out
