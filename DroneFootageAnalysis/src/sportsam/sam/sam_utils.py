import cv2
import matplotlib.pyplot as plt
import numpy as np
import subprocess

import torch
from pathlib import Path

# TODO return bool useful?


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
    output_pattern = output_path / f"%05d.jpg"

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


def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    Show a mask on a matplotlib axis with a specific color.

    Args:
        mask: A binary mask of shape (H, W) or (1, H, W).
        ax: The matplotlib axis to draw on.
        obj_id: Optional object ID for color mapping.
        random_color: If True, use a random color instead of a fixed one.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def add_masks_to_frame(frame_path, masks_dict, output_path, alpha=0.5):
    """
    Add colored masks to a frame and save as JPG

    Args:
        frame_path: Path to the original frame JPG
        masks_dict: Dictionary of {obj_id: mask_array} for this frame
        output_path: Where to save the result
        alpha: Transparency of the overlay (0.0 = transparent, 1.0 = opaque)
    """
    frame = cv2.imread(frame_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay = np.zeros_like(frame_rgb)

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]

    for _, (obj_id, mask) in enumerate(masks_dict.items()):
        # Fix the mask shape
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Remove first dimension if it's 1
        elif len(mask.shape) == 3:
            mask = mask[0]  # Take first slice if multiple

        # Verify shape matches frame
        if mask.shape != frame_rgb.shape[:2]:
            print(
                f"Warning: Mask shape {mask.shape} doesn't match frame {frame_rgb.shape[:2]}"
            )
            continue

        color = colors[obj_id % len(colors)]
        overlay[mask] = color

    result = cv2.addWeighted(frame_rgb, 1 - alpha, overlay, alpha, 0)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)


def add_masks_to_blank(frame_size, masks_dict, output_path, alpha=1):
    """
    Add colored masks to a frame and save as JPG

    Args:
        frame_size: Size of the frame as (height, width)
        masks_dict: Dictionary of {obj_id: mask_array} for this frame
        output_path: Where to save the result
        alpha: Transparency of the overlay (0.0 = transparent, 1.0 = opaque)
    """
    # TODO frame and height in the input seem to be flipped?
    frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
    overlay = np.zeros_like(frame)

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]

    for obj_id, mask in masks_dict.items():
        # Fix the mask shape
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Remove first dimension if it's 1
        elif len(mask.shape) == 3:
            mask = mask[0]  # Take first slice if multiple

        # Verify shape matches frame
        if mask.shape != frame.shape[:2]:
            print(
                f"Warning: Mask shape {mask.shape} doesn't match frame {frame.shape[:2]}"
            )
            continue

        color = colors[obj_id % len(colors)]
        overlay[mask] = color

    # TODO addWeighted seems to still not give solid masks, even with alpha=1
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)


def make_dir(base_path: Path):
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
