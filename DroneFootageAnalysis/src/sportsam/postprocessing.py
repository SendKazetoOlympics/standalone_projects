"""
postprocessing.py

Contains functions for postprocessing, statistics, and visualization of segmentation results,
including mask overlays, area/moment calculations, graphing, and path tracking.
"""

import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# from torchvision.io import read_image


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

    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)


def create_speed_graph(first_moments, timestamps, output_file):
    if len(first_moments) < 2:
        print("Not enough points to calculate speed")
        return

    speeds = []
    time_points = []

    for i in range(1, len(first_moments)):
        x1, y1 = first_moments[i - 1]
        x2, y2 = first_moments[i]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        time_diff = timestamps[i] - timestamps[i - 1]
        if time_diff > 0:
            speed = distance / time_diff
            speeds.append(speed)
            time_points.append(timestamps[i])

    if not speeds:
        print("Could not calculate speeds")
        return

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_points, speeds, "b-", linewidth=2, label="Speed")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Speed (pixels/second)")
    plt.title("Object Speed Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)

    stats_text = f"Avg: {avg_speed:.1f} px/s\nMax: {max_speed:.1f} px/s\nMin: {min_speed:.1f} px/s"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.subplot(1, 2, 2)
    x_coords = [point[0] for point in first_moments]
    y_coords = [point[1] for point in first_moments]
    colors = []
    colors.append(speeds[0] if speeds else 0)
    colors.extend(speeds)
    scatter = plt.scatter(x_coords, y_coords, c=colors, cmap="viridis", s=20, alpha=0.7)
    plt.plot(x_coords, y_coords, "k-", alpha=0.3, linewidth=1)
    plt.colorbar(scatter, label="Speed (px/s)")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.title("Object Path (colored by speed)")
    plt.gca().invert_yaxis()
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Speed graph saved as: {output_file}")
    print(f"Average speed: {avg_speed:.2f} pixels/second")
    print(f"Maximum speed: {max_speed:.2f} pixels/second")
    print(f"Minimum speed: {min_speed:.2f} pixels/second")

    return speeds, time_points


def create_graph(x, y, x_axis, y_axis, title, output_file):
    """
    Create a graph from x and y data and save it as an image.

    Args:
        x (list): X-axis data.
        y (list): Y-axis data.
        x_axis (str): Label for the x-axis.
        y_axis (str): Label for the y-axis.
        title (str): Title of the graph.
        output_file (Path): Path to save the graph image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.grid(True)
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved as: {output_file}")


def track_object_path(video_path, output_path):
    """
    Track an object in a video and visualize its path with speed calculation.

    Args:
        video_path (Path): Path to the input video file.
        output_path (Path): Path to save the output video with tracking.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output video will be saved to: {str(output_path)}")

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return

    areas = []
    first_moments = []
    second_moments = []
    timestamps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask_path = video_path.parent / f"mask_tensors/{frame_count:05d}_1_mask.pt"
        area = zeroth_image_moment(mask_path)
        areas.append(area)
        first_moment = first_image_moment(mask_path)
        first_moments.append(first_moment)
        second_moment = second_image_moment(mask_path)
        second_moments.append(second_moment)
        timestamps.append(frame_count / fps)

        if len(first_moments) > 1:
            for i in range(1, len(first_moments)):
                cv2.line(
                    frame,
                    first_moments[i - 1],
                    first_moments[i],
                    (0, 255, 255),
                    3,
                )

        cv2.circle(frame, first_moment, 5, (0, 0, 255), -1)

        if output_path:
            out.write(frame)

        frame_count += 1

    cap.release()
    if output_path:
        out.release()

    area_args = [
        timestamps,
        areas,
        "Time (seconds)",
        "Area (pixels)",
        "Area Over Time",
    ]
    create_graph(*area_args, output_file=output_path.parent / "area_graph.png")

    x_args = [
        timestamps,
        [point[0] for point in first_moments],
        "Time (seconds)",
        "X Position (pixels)",
        "X Position Over Time",
    ]
    create_graph(
        *x_args,
        output_file=output_path.parent / "x_graph.png",
    )

    y_args = [
        timestamps,
        [point[1] for point in first_moments],
        "Time (seconds)",
        "Y Position (pixels)",
        "Y Position Over Time",
    ]
    create_graph(
        *y_args,
        output_file=output_path.parent / "y_graph.png",
    )

    xx_args = [
        timestamps,
        [point[0] for point in second_moments],
        "Time (seconds)",
        "Ixx",
        "Ixx Over Time",
    ]
    create_graph(
        *xx_args,
        output_file=output_path.parent / "ixx_graph.png",
    )

    yy_args = [
        timestamps,
        [point[1] for point in second_moments],
        "Time (seconds)",
        "Iyy",
        "Iyy Over Time",
    ]
    create_graph(
        *yy_args,
        output_file=output_path.parent / "iyy_graph.png",
    )

    xy_args = [
        timestamps,
        [point[2] for point in second_moments],
        "Time (seconds)",
        "Ixy",
        "Ixy Over Time",
    ]
    create_graph(
        *xy_args,
        output_file=output_path.parent / "ixy_graph.png",
    )

    create_speed_graph(
        first_moments, timestamps, output_path.parent / "speed_graph.png"
    )


def convert_mask_to_bounding_box(mask_path):
    """
    Convert a binary mask to a bounding box and save it as an image.

    Args:
        mask_path (Path): Path to the input binary mask image.
    """
    # from torchvision.io import read_image
    mask = torch.load(mask_path)
    if mask is None:
        print(f"Error: Could not read mask from {mask_path}")
        return
    obj_ids = torch.unique(mask)
    print(obj_ids)
    print(mask.size())
    print(mask)
