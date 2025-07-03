"""
io_handler.py

Handles input/output operations such as extracting frames from video, creating videos from frames,
directory management, and CSV file output.
"""

import csv
import shutil
from pathlib import Path

import decord
import imageio
import torch
from jaxtyping import Int, Bool


class IOHandler:
    temp_dir: Path
    output_dir: Path

    def __init__(self, temp_dir: Path | str, output_dir: Path | str):
        """Initialize IOHandler with temporary and output directories.
        Args:
            temp_dir (Path | str): Path to the temporary directory for storing extracted frames.
            output_dir (Path | str): Path to the output directory where results will be saved.

        """
        if isinstance(temp_dir, str):
            temp_dir = Path(temp_dir)
        self.temp_dir = Path(temp_dir)

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = Path(output_dir)

        if not self.temp_dir.exists():
            raise FileNotFoundError(
                f"Temporary directory {self.temp_dir} does not exist. Please create it before using IOHandler."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO different sized frames? Probably raise error if not all the same width/height
    def extract_frames_from_videos(self, videos: list[str]) -> None:
        """Extract frames from a list of video files or directories containing JPEG images.
        Args:
            video_paths (list[str]): List of paths to video files or directories containing JPEG images.

        """
        if len(videos) == 0:
            raise ValueError("No video paths provided for frame extraction.")

        frame_count = 0
        batch_size = 250
        batch_num = 0
        batch_frame_count = 0

        def get_batch_dir(batch_num):
            batch_dir = self.temp_dir / f"batch{batch_num}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            return batch_dir

        batch_dir = get_batch_dir(batch_num)

        for video in videos:
            video_path = Path(video)

            if not video_path.exists():
                raise FileNotFoundError(f"Video file {video_path} not found.")

            if video_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
                vr = decord.VideoReader(str(video_path))
                for frame in vr:
                    if batch_frame_count >= batch_size:
                        batch_num += 1
                        batch_frame_count = 0
                        batch_dir = get_batch_dir(batch_num)
                    imageio.imwrite(
                        batch_dir / f"{frame_count:05d}.jpg", frame.asnumpy()
                    )
                    frame_count += 1
                    batch_frame_count += 1
            elif video_path.is_dir():
                frames = sorted(
                    [
                        p
                        for p in video_path.iterdir()
                        if p.suffix in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                    ]
                )
                for frame in frames:
                    if batch_frame_count >= batch_size:
                        batch_num += 1
                        batch_frame_count = 0
                        batch_dir = get_batch_dir(batch_num)
                    shutil.copy(frame, batch_dir / f"{frame_count:05d}.jpg")
                    frame_count += 1
                    batch_frame_count += 1
            else:
                raise ValueError(
                    f"Unsupported video file format: {video_path.suffix}. Please provide a video file with .mp4, .avi, or .mov extension."
                )

            print(f"Extracted frames from {video_path}...")

    def extract_frames_from_manifest(self, manifest: str) -> None:
        """Extract frames from a list of video files specified in a manifest file.
        Args:
            manifest (Path): Path to the CSV manifest file containing video file paths.

        """
        manifest_path = Path(manifest)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file {manifest} not found.")

        videos = []

        if not manifest_path.is_file():
            raise ValueError(f"Manifest {manifest} is not a valid file.")
        if manifest_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Unsupported manifest file format: {manifest_path.suffix}. Please provide a CSV file."
            )

        with open(manifest, "r") as f:
            reader = csv.reader(f)
            videos = [row[0] for row in reader if row]

        self.extract_frames_from_videos(videos)

    def save_output_masks(
        self, results: dict[Int, dict[Int, Bool[torch.Tensor, "H W"]]]
    ) -> None:
        for frame_idx, masks_dict in results.items():
            for obj_id, mask in masks_dict.items():
                torch.save(
                    torch.tensor(mask, dtype=torch.uint8),
                    self.output_dir / f"masks/{frame_idx:05d}_{obj_id}_mask.pt",
                )

    def save_inference_state(self):
        raise NotImplementedError

    def load_inference_state(self):
        raise NotImplementedError

    # TODO change according to what's next for analysis.py
    def create_graph(self, data: list[tuple[Int, Bool[torch.Tensor, "H W"]]]) -> None:
        raise NotImplementedError


##### TODO refactor everything below #####


# def create_video_from_frames(input_dir, frame_format, output_file, framerate):
#     """
#     Create MP4 video from sequence of JPEG images using ffmpeg.

#     Args:
#         input_dir (str): Directory containing input JPEG frames
#         frame_format (str): Format of the input frames (e.g., "%05d.jpg")
#         output_file (str): Path to the output video file
#         framerate (float): Frames per second for the output video

#     Returns:
#         bool: True if successful, False otherwise

#     """
#     # Ensure input directory exists
#     input_path = Path(input_dir)
#     if not input_path.exists():
#         print(f"Error: Input directory '{input_dir}' not found.")
#         return False

#     # Check if input images exist
#     frame_files = list(input_path.glob("*.jpg"))
#     if not frame_files:
#         print(f"Error: No JPEG files found in '{input_dir}'.")
#         return False

#     # Construct input pattern for ffmpeg
#     input_pattern = input_path / frame_format

#     # Construct ffmpeg command
#     cmd = [
#         "ffmpeg",
#         "-framerate",
#         str(framerate),
#         "-i",
#         str(input_pattern),
#         "-c:v",
#         "libx264",
#         "-pix_fmt",
#         "yuv420p",
#         "-y",  # Overwrite output file if it exists
#         str(output_file),
#     ]

#     try:
#         print(f"Creating video from frames in '{input_dir}'...")
#         print(f"Command: {' '.join(cmd)}")

#         # Run ffmpeg command
#         _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

#         print(f"Successfully created video: '{output_file}'")
#         print(f"Used {len(frame_files)} frames at {framerate} fps")
#         return True

#     except subprocess.CalledProcessError as e:
#         print(f"Error running ffmpeg: {e}")
#         print(f"stderr: {e.stderr}")
#         return False
#     except FileNotFoundError:
#         print(
#             "Error: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH."
#         )
#         return False


# def make_dir(base_path: Path):
#     """
#     Create a directory at the given path, or a new one with an incremented suffix if it exists.

#     Args:
#         base_path (Path): The base directory path to create.

#     Returns:
#         Path: The created directory path.
#     """
#     if not base_path.exists():
#         base_path.mkdir(parents=True, exist_ok=True)
#         print(f"Created directory: {base_path}")
#         return base_path

#     parent = base_path.parent
#     stem = base_path.name
#     i = 1
#     while True:
#         new_path = parent / f"{stem}{i}"
#         if not new_path.exists():
#             new_path.mkdir(parents=True, exist_ok=True)
#             print(f"Created directory: {new_path}")
#             return new_path
#         i += 1


# def create_csv(x, y, x_axis, y_axis, output_file):
#     """
#     Create a CSV file from the provided data.

#     Args:
#         x (list): X-axis data.
#         y (list): Y-axis data.
#         x_axis (str): Label for the x-axis.
#         y_axis (str): Label for the y-axis.
#         output_file (Path): Path to save the CSV file.
#     """
#     data = {x_axis: x, y_axis: y}
#     df = pd.DataFrame(data)
#     df.to_csv(output_file, index=False)
#     print(f"CSV file saved as: {output_file}")
