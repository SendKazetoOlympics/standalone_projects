"""
io_handler.py

Handles input/output operations such as extracting frames from video, creating videos from frames,
directory management, and CSV file output.
"""

import csv
import shutil
from pathlib import Path

import cv2
import decord
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int, Bool


class IOHandler:
    temp_dir: Path
    output_dir: Path
    videos: dict[Int, Int]

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
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        self.videos = {}

    def batch_frames(self, batch_size: int = 250) -> None:
        """Organize extracted frames into batches.
        Args:
            batch_size (int, optional): Number of frames per batch. Defaults to 250.
        """
        frame_files = sorted(list(self.temp_dir.glob("*.jpg")))
        if not frame_files:
            print("No frames found to batch.")
            return

        batch_num = 0
        for i in range(0, len(frame_files), batch_size):
            batch_dir = self.temp_dir / f"batch{batch_num}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            batch_files = frame_files[i : i + batch_size]
            for frame_file in batch_files:
                shutil.move(frame_file, batch_dir / frame_file.name)

            batch_num += 1

    def unbatch_frames(self) -> None:
        """Move all frames from batch directories back to temp_dir."""
        batch_dirs = sorted(
            [
                d
                for d in self.temp_dir.iterdir()
                if d.is_dir() and d.name.startswith("batch")
            ]
        )

        if not batch_dirs:
            print("No batch directories found to unbatch.")
            return

        # Move all frames back to temp_dir
        for batch_dir in batch_dirs:
            frames = list(batch_dir.glob("*.jpg"))
            for frame in frames:
                shutil.move(frame, self.temp_dir / frame.name)

            # Remove empty batch directory
            batch_dir.rmdir()

        print(f"Successfully unbatched frames from {len(batch_dirs)} directories.")

    # TODO different sized frames? Probably raise error if not all the same width/height
    def extract_frames_from_videos(self, videos: list[str]) -> None:
        """Extract frames from a list of video files or directories containing JPEG images.
        Args:
            videos (list[str]): List of paths to video files or directories containing JPEG images.

        Raises:
            ValueError: If no video paths are provided or if video format is unsupported.
            FileNotFoundError: If video file is not found.
        """
        if len(videos) == 0:
            raise ValueError("No video paths provided for frame extraction.")

        frame_count = 0
        video_idx = 0
        for video in videos:
            video_path = Path(video)

            if not video_path.exists():
                raise FileNotFoundError(f"Video file {video_path} not found.")

            if video_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
                vr = decord.VideoReader(str(video_path))
                for frame in vr:
                    imageio.imwrite(
                        self.temp_dir / f"{frame_count:05d}.jpg", frame.asnumpy()
                    )
                    frame_count += 1
            elif video_path.is_dir():
                frames = sorted(
                    [
                        p
                        for p in video_path.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg"}
                    ]
                )
                for frame in frames:
                    shutil.copy(frame, self.temp_dir / f"{frame_count:05d}.jpg")
                    frame_count += 1
            else:
                raise ValueError(
                    f"Unsupported video file format: {video_path.suffix}. Please provide a video file with .mp4, .avi, or .mov extension."
                )

            self.videos[video_idx] = frame_count - sum(self.videos)
            video_idx += 1

        self.batch_frames()
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
                    self.output_dir / f"/{frame_idx:05d}_{obj_id}_mask.pt",
                )

    def group_frames_by_video(self) -> None:
        # TODO group mask frames by video from self.videos
        # name each one by video{video_idx}, maybe later have original titles
        # structure for each video:
        # masks/
        # visualization/
        # graphs/
        # csvs/
        # output video
        raise NotImplementedError

    def recreate_video_from_frames_dir(self, video_dir: Path) -> None:
        # TODO input path
        raise NotImplementedError

    def save_inference_state(self):
        raise NotImplementedError

    def load_inference_state(self):
        raise NotImplementedError

    # TODO rethink implementation
    def write_centroid(
        self,
        first_moment: dict[Int, tuple[Int, Int]],
        video_idx: Int,
        obj_id: Int = 1,
    ) -> None:
        """Write object centroid position over time from first image moments.
        For each frame, overlays the segmentation mask and draws tracking visualization
        showing centroid position and movement trail.

        Args:
            first_moment (dict[Int, tuple[Int, Int]]): Dictionary mapping frame indices to (x,y) centroid coordinates,
                as returned by Analyzer.first_image_moment()
            video_idx (Int): Index of the video in videos dict.
            obj_id (Int): ID of the object you want to track.
        """

        # Sort by frame index to ensure temporal ordering
        tracking = []
        for frame_idx in sorted(first_moment.keys()):
            x, y = first_moment[frame_idx]
            tracking.append((frame_idx, x, y))

        # TODO leave this here?
        # Create visualization directory
        vis_dir = self.output_dir / f"video{video_idx}/visualization"
        vis_dir.mkdir(exist_ok=True)

        # For storing centroid trail
        trail = []

        # Process each frame
        for frame_idx, x, y in tracking:
            # Load original frame; temp_dir MUST BE UNBATCHED
            frame_path = self.temp_dir / f"{frame_idx:05d}.jpg"
            if not frame_path.exists():
                continue
            frame = cv2.imread(str(frame_path))

            # Load and overlay mask if it exists
            mask_path = (
                self.output_dir
                / f"video{video_idx}/masks/{frame_idx:05d}_{obj_id}_mask.pt"
            )
            if mask_path.exists():
                mask = torch.load(mask_path).numpy().astype(np.uint8)
                mask_overlay = np.zeros_like(frame)
                mask_overlay[mask == 1] = [0, 255, 0]  # Green overlay
                frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.3, 0)

            # Draw centroid trail
            trail.append((x, y))
            if len(trail) > 10:
                # Draw line connecting previous centroids
                points = np.array(trail[-10:], dtype=np.int32)  # Keep last 10 points
                cv2.polylines(frame, [points], False, (0, 0, 255), 2)

            # Draw current centroid position
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue dot

            # Save visualization
            vis_path = vis_dir / f"{frame_idx:05d}_tracked.jpg"
            cv2.imwrite(str(vis_path), frame)

    @staticmethod
    def create_graph(
        graph_dir: Path,
        title: str,
        x_data: list[Int],
        x_axis_title: str,
        y_data: list[Int],
        y_axis_title: str,
    ) -> None:
        """Create a plot of a given data set."""
        # x_axis (str): Label for the x-axis.
        # y_axis (str): Label for the y-axis.
        # title (str): Title of the graph.

        plt.figure(figsize=(10, 5))
        plt.plot(x_data, y_data, marker="o", linestyle="-")
        plt.xlabel(x_axis_title)
        plt.ylabel(y_axis_title)
        plt.title(title)
        plt.grid(True)
        plt.savefig(str(graph_dir / f"{title}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{title} graph saved in: {graph_dir}")


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
