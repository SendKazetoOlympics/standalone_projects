"""
io_handler.py

Handles input/output operations such as extracting frames from video, creating videos from frames,
directory management, and CSV file output.
"""

import subprocess
from pathlib import Path
import pandas as pd

class IOHandler:
    
    def __init__(self):
        pass

    def extract_frames_from_videos(self, video_path: list[str]):
        raise NotImplementedError("Method not implemented yet.")

    def extract_frames_from_file_list(self, manifest: str):
        raise NotImplementedError("Method not implemented yet.")

def extract_frames_from_video(video_path, quality=2):
    """
    Extract JPEG frames from a video file using ffmpeg.

    Args:
        video_path (Path): Path to the input video file
        quality (int): JPEG quality (1-31, lower is better quality)

    Returns:
        int: Number of frames extracted, or False if an error occurred
    """

    # Ensure input directory exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    # Ensure output directory exists
    output_path = video_path.parent / video_path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    # Construct ffmpeg command
    output_pattern = output_path / "%05d.jpg"

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-q:v",
        str(quality),
        "-start_number",
        "0",
        str(output_pattern),
    ]

    try:
        print(f"Extracting frames from '{video_path}' to '{output_path}'...")
        print(f"Command: {' '.join(cmd)}")

        # Run ffmpeg command
        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count extracted frames
        frame_count = len(list(output_path.glob("*.jpg")))
        print(f"Successfully extracted {frame_count} frames to '{output_path}'")

        return frame_count

    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(
            "Error: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH."
        )
        return False


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
