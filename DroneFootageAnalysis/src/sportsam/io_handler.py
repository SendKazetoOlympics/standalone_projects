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
import matplotlib.pyplot as plt
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
        """Group mask files by video and create organized directory structure."""
        if not self.videos:
            print("No video mapping found. Cannot group frames by video.")
            return

        # Get all mask files from the main masks directory
        masks_dir = self.output_dir

        mask_files = list(masks_dir.glob("*_mask.pt"))
        if not mask_files:
            print("No mask files found to group.")
            return

        # Calculate frame ranges for each video
        frame_ranges = {}
        current_frame = 0
        for video_idx, frame_count in self.videos.items():
            frame_ranges[video_idx] = (current_frame, current_frame + frame_count - 1)
            current_frame += frame_count

        # Create directory structure for each video
        for video_idx in self.videos.keys():
            video_dir = self.output_dir / f"video{video_idx}"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (video_dir / "masks").mkdir(parents=True, exist_ok=True)
            (video_dir / "graphs").mkdir(parents=True, exist_ok=True)
            (video_dir / "csvs").mkdir(parents=True, exist_ok=True)
            (video_dir / "visualization").mkdir(parents=True, exist_ok=True)

        # Move mask files to appropriate video directories
        for mask_file in mask_files:
            # Extract frame index from filename (format: {frame_idx:05d}_{obj_id}_mask.pt)
            filename = mask_file.name
            try:
                frame_idx = int(filename.split("_")[0])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse frame index from {filename}")
                continue

            # Find which video this frame belongs to
            target_video = None
            for video_idx, (start_frame, end_frame) in frame_ranges.items():
                if start_frame <= frame_idx <= end_frame:
                    target_video = video_idx
                    break

            if target_video is not None:
                # Move mask file to video-specific masks directory
                target_dir = self.output_dir / f"video{target_video}" / "masks"
                target_path = target_dir / filename
                shutil.move(str(mask_file), str(target_path))
            else:
                print(f"Warning: Frame {frame_idx} does not belong to any video range")

        print(f"Successfully grouped masks into {len(self.videos)} video directories")

    # TODO this is broken since I can't import cv2 in this file for some reason?!?!
    # @staticmethod
    # def recreate_video_from_frames_dir(
    #     video_dir: Path,
    #     framerate: float = 30.0,  # TODO save framerate somewhere?
    #     output_filename: str = "output_video.mp4",
    # ) -> bool:
    #     """Create MP4 video from sequence of JPEG images in a directory using OpenCV.
    #
    #     Args:
    #         video_dir (Path): Directory containing input JPEG frames
    #         framerate (float): Frames per second for the output video. Defaults to 30.0.
    #         output_filename (str): Name of the output video file. Defaults to "output_video.mp4".
    #
    #     Returns:
    #         bool: True if successful, False otherwise
    #
    #     Raises:
    #         FileNotFoundError: If video_dir does not exist or contains no JPEG files
    #     """
    #     # Ensure input directory exists
    #     if not video_dir.exists():
    #         raise FileNotFoundError(f"Video directory '{video_dir}' not found.")
    #
    #     if not video_dir.is_dir():
    #         raise ValueError(f"Path '{video_dir}' is not a directory.")
    #
    #     # Check if input images exist
    #     frame_files = list(video_dir.glob("*.jpg"))
    #     if not frame_files:
    #         raise FileNotFoundError(f"No JPEG files found in '{video_dir}'.")
    #
    #     # Sort frame files to ensure proper order
    #     frame_files.sort()
    #
    #     # Get output file path
    #     output_file = video_dir / output_filename
    #
    #     try:
    #         print(f"Creating video from frames in '{video_dir}'...")
    #
    #         # Read first frame to get dimensions
    #         first_frame = cv2.imread(str(frame_files[0]))
    #         if first_frame is None:
    #             print(f"Error: Could not read first frame '{frame_files[0]}'")
    #             return False
    #
    #         height, width, _channels = first_frame.shape
    #
    #         # Define the codec and create VideoWriter object
    #         fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    #         video_writer = cv2.VideoWriter(
    #             str(output_file), fourcc, framerate, (width, height)
    #         )
    #
    #         if not video_writer.isOpened():
    #             print("Error: Could not open video writer")
    #             return False
    #
    #         # Write each frame to the video
    #         frames_written = 0
    #         for frame_file in frame_files:
    #             frame = cv2.imread(str(frame_file))
    #             if frame is None:
    #                 print(f"Warning: Could not read frame '{frame_file}', skipping...")
    #                 continue
    #
    #             # Ensure frame has correct dimensions
    #             if frame.shape[:2] != (height, width):
    #                 print(
    #                     f"Warning: Frame '{frame_file}' has different dimensions, resizing..."
    #                 )
    #                 frame = cv2.resize(frame, (width, height))
    #
    #             video_writer.write(frame)
    #             frames_written += 1
    #
    #         # Release the video writer
    #         video_writer.release()
    #
    #         print(f"Successfully created video: '{output_file}'")
    #         print(f"Used {frames_written} frames at {framerate} fps")
    #         return True
    #
    #     except Exception as e:
    #         print(f"Error creating video: {e}")
    #         return False

    def save_inference_state(self):
        raise NotImplementedError

    def load_inference_state(self):
        raise NotImplementedError

    # TODO this is broken since I can't import cv2 in this file for some reason?!?!
    # TODO this definitely needs to be refactored
    # def write_centroid(
    #     self,
    #     first_moment: dict[Int, tuple[Int, Int]],
    #     video_idx: Int,
    #     obj_id: Int = 1,
    # ) -> None:
    #     """Write object centroid position over time from first image moments.
    #     For each frame, overlays the segmentation mask and draws tracking visualization
    #     showing centroid position and movement trail.
    #
    #     Args:
    #         first_moment (dict[Int, tuple[Int, Int]]): Dictionary mapping frame indices to (x,y) centroid coordinates,
    #             as returned by Analyzer.first_image_moment()
    #         video_idx (Int): Index of the video in videos dict.
    #         obj_id (Int): ID of the object you want to track.
    #     """
    #
    #     # Sort by frame index to ensure temporal ordering
    #     tracking = []
    #     for frame_idx in sorted(first_moment.keys()):
    #         x, y = first_moment[frame_idx]
    #         tracking.append((frame_idx, x, y))
    #
    #     # TODO leave this here? It's already done in group_frames_by_video
    #     # Create visualization directory
    #     vis_dir = self.output_dir / f"video{video_idx}/visualization"
    #     vis_dir.mkdir(exist_ok=True)
    #
    #     # For storing centroid trail
    #     trail = []
    #
    #     # Process each frame
    #     for frame_idx, x, y in tracking:
    #         # Load original frame; temp_dir MUST BE UNBATCHED
    #         frame_path = self.temp_dir / f"{frame_idx:05d}.jpg"
    #         if not frame_path.exists():
    #             continue
    #         frame = cv2.imread(str(frame_path))
    #
    #         # Load and overlay mask if it exists
    #         mask_path = (
    #             self.output_dir
    #             / f"video{video_idx}/masks/{frame_idx:05d}_{obj_id}_mask.pt"
    #         )
    #         if mask_path.exists():
    #             mask = torch.load(mask_path).numpy().astype(np.uint8)
    #             mask_overlay = np.zeros_like(frame)
    #             mask_overlay[mask == 1] = [0, 255, 0]  # Green overlay
    #             frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.3, 0)
    #
    #         # Draw centroid trail
    #         trail.append((x, y))
    #         if len(trail) > 10:
    #             # Draw line connecting previous centroids
    #             points = np.array(trail[-10:], dtype=np.int32)  # Keep last 10 points
    #             cv2.polylines(frame, [points], False, (0, 0, 255), 2)
    #
    #         # Draw current centroid position
    #         cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue dot
    #
    #         # Save visualization
    #         vis_path = vis_dir / f"{frame_idx:05d}_tracked.jpg"
    #         cv2.imwrite(str(vis_path), frame)

    @staticmethod
    def create_graph(
        output_dir: Path,
        title: str,
        x_data: list[Int],
        x_axis_title: str,
        y_data: list[Int],
        y_axis_title: str,
    ) -> None:
        """Create a plot of a given data set and save it as a PNG image.

        Args:
            output_dir (Path): Directory where the graph image will be saved.
            title (str): Title of the graph, also used as the filename (with .png extension).
            x_data (list[Int]): Data points for the x-axis.
            x_axis_title (str): Label for the x-axis.
            y_data (list[Int]): Data points for the y-axis.
            y_axis_title (str): Label for the y-axis.
        """

        plt.figure(figsize=(10, 5))
        plt.plot(x_data, y_data, marker="o", linestyle="-")
        plt.xlabel(x_axis_title)
        plt.ylabel(y_axis_title)
        plt.title(title)
        plt.grid(True)
        plt.savefig(str(output_dir / f"{title}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{title} graph saved in: {output_dir}")

    @staticmethod
    def convert_data_dict_to_list(
        data_dict: dict, sort_by_key: bool = True, tuple_index: int | None = None
    ) -> tuple[list, list]:
        """Convert a dictionary of data to a list format for easier processing.

        Args:
            data_dict (dict): Dictionary to convert. Can handle various formats:
                - dict[int, int]: frame_idx -> value (e.g., areas)
                - dict[int, tuple]: frame_idx -> (x, y) (e.g., centroids)
                - dict[int, tuple]: frame_idx -> (xx, yy, xy) (e.g., second moments)
            sort_by_key (bool): Whether to sort by dictionary keys. Defaults to True.
            tuple_index (int | None): If provided, extract this index from tuple values.
                For example, tuple_index=0 extracts x-coordinates from centroids.
                Defaults to None (return full tuples).

        Returns:
            tuple[list, list]: (keys_list, values_list) where:
                - keys_list: List of dictionary keys (typically frame indices)
                - values_list: List of dictionary values (scalars, tuples, or extracted elements)

        Examples:
            >>> areas = {0: 100, 1: 120, 2: 90}
            >>> keys, values = IOHandler.convert_data_dict_to_list(areas)
            >>> # keys = [0, 1, 2], values = [100, 120, 90]

            >>> centroids = {0: (10, 20), 1: (15, 25), 2: (12, 18)}
            >>> keys, values = IOHandler.convert_data_dict_to_list(centroids)
            >>> # keys = [0, 1, 2], values = [(10, 20), (15, 25), (12, 18)]

            >>> # Extract x-coordinates only
            >>> keys, x_coords = IOHandler.convert_data_dict_to_list(centroids, tuple_index=0)
            >>> # keys = [0, 1, 2], x_coords = [10, 15, 12]

            >>> # Extract y-coordinates only
            >>> keys, y_coords = IOHandler.convert_data_dict_to_list(centroids, tuple_index=1)
            >>> # keys = [0, 1, 2], y_coords = [20, 25, 18]
        """
        if not data_dict:
            return [], []

        # Get items from dictionary
        items = list(data_dict.items())

        # Sort by key if requested
        if sort_by_key:
            items.sort(key=lambda x: x[0])

        # Separate keys and values
        keys = [item[0] for item in items]
        values = [item[1] for item in items]

        # Extract specific tuple element if requested
        if tuple_index is not None:
            extracted_values = []
            for value in values:
                if isinstance(value, (tuple, list)):
                    try:
                        extracted_values.append(value[tuple_index])
                    except IndexError:
                        raise IndexError(
                            f"tuple_index {tuple_index} is out of range for tuple {value}"
                        )
                else:
                    raise TypeError(
                        f"tuple_index provided but value {value} is not a tuple or list"
                    )
            values = extracted_values

        return keys, values


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
