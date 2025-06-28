"""
io_handler.py

Handles input/output operations such as extracting frames from video, creating videos from frames,
directory management, and CSV file output.
"""

import subprocess
from pathlib import Path
import tempfile
import shutil
import decord
import imageio
import csv


class IOHandler:

    def __init__(self):
        pass

    def extract_frames_from_videos(self, video_paths: list[str]) -> Path:
        """Extract frames from a list of video files or directories containing JPEG images.
        Args:
            video_paths (list[str]): List of paths to video files or directories containing JPEG images.

        Returns:
            Path: Path to the temporary directory containing extracted frames.

        """
        if video_paths is None or len(video_paths) == 0:
            raise ValueError("No video paths provided for frame extraction.")

        temp_dir = Path(tempfile.mkdtemp())

        frame_count = 0

        for video_path in video_paths:
            video_path_obj = Path(video_path)

            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video file {video_path} not found.")
            if not video_path_obj.is_file():
                raise ValueError(
                    f"Unsupported video file format: {video_path_obj.suffix}. Please provide either a video file or a directory containing JPEG images."
                )

            if video_path_obj.suffix.lower() in {".mp4", ".avi", ".mov"}:
                vr = decord.VideoReader(video_path_obj)
                for frame in vr:
                    imageio.imwrite(
                        temp_dir / f"{frame_count:05d}.jpg", frame.asnumpy()
                    )
                    frame_count += 1
            elif video_path_obj.is_dir():
                frames = sorted(
                    [
                        p
                        for p in video_path_obj.iterdir()
                        if p.suffix in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                    ]
                )
                for frame in frames:
                    shutil.copy(frame, temp_dir / f"{frame_count:05d}.jpg")
                    frame_count += 1
            else:
                raise ValueError(
                    f"Unsupported video file format: {video_path_obj.suffix}. Please provide a video file with .mp4, .avi, or .mov extension."
                )

            print(f"Extracted frames from {video_path}...")

        return temp_dir

    def extract_frames_from_file_list(self, manifest: str):
        raise NotImplementedError("Method not implemented yet.")

    def clear_tmp(self, temp_dir: Path) -> None:
        """Clear the temporary directory where frames are stored."""
        if temp_dir.exists() and temp_dir.is_dir():
            shutil.rmtree(temp_dir)
        else:
            print(
                f"Temporary directory {temp_dir} does not exist or is not a directory."
            )


##### TODO refactor everything below #####


def create_video_from_frames(input_dir, frame_format, output_file, framerate):
    """
    Create MP4 video from sequence of JPEG images using ffmpeg.

    Args:
        input_dir (str): Directory containing input JPEG frames
        frame_format (str): Format of the input frames (e.g., "%05d.jpg")
        output_file (str): Path to the output video file
        framerate (float): Frames per second for the output video

    Returns:
        bool: True if successful, False otherwise

    """
    # Ensure input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' not found.")
        return False

    # Check if input images exist
    frame_files = list(input_path.glob("*.jpg"))
    if not frame_files:
        print(f"Error: No JPEG files found in '{input_dir}'.")
        return False

    # Construct input pattern for ffmpeg
    input_pattern = input_path / frame_format

    # Construct ffmpeg command
    cmd = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        str(input_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",  # Overwrite output file if it exists
        str(output_file),
    ]

    try:
        print(f"Creating video from frames in '{input_dir}'...")
        print(f"Command: {' '.join(cmd)}")

        # Run ffmpeg command
        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"Successfully created video: '{output_file}'")
        print(f"Used {len(frame_files)} frames at {framerate} fps")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(
            "Error: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH."
        )
        return False


def make_dir(base_path: Path):
    """
    Create a directory at the given path, or a new one with an incremented suffix if it exists.

    Args:
        base_path (Path): The base directory path to create.

    Returns:
        Path: The created directory path.
    """
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {base_path}")
        return base_path

    parent = base_path.parent
    stem = base_path.name
    i = 1
    while True:
        new_path = parent / f"{stem}{i}"
        if not new_path.exists():
            new_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {new_path}")
            return new_path
        i += 1


def create_csv(x, y, x_axis, y_axis, output_file):
    """
    Create a CSV file from the provided data.

    Args:
        x (list): X-axis data.
        y (list): Y-axis data.
        x_axis (str): Label for the x-axis.
        y_axis (str): Label for the y-axis.
        output_file (Path): Path to save the CSV file.
    """
    data = {x_axis: x, y_axis: y}
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CSV file saved as: {output_file}")
