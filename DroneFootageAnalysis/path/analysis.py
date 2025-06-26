import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import pandas as pd

# from torchvision.ops import masks_to_boxes
from torchvision.io import read_image


# TODO refactor to create_graph
def create_speed_graph(first_moments, timestamps, output_file):
    if len(first_moments) < 2:
        print("Not enough points to calculate speed")
        return

    speeds = []
    time_points = []

    # Calculate speed between consecutive points
    for i in range(1, len(first_moments)):
        # Calculate distance between points (in pixels)
        x1, y1 = first_moments[i - 1]
        x2, y2 = first_moments[i]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate time difference
        time_diff = timestamps[i] - timestamps[i - 1]

        # Calculate speed (pixels per second)
        if time_diff > 0:
            speed = distance / time_diff
            speeds.append(speed)
            time_points.append(timestamps[i])

    if not speeds:
        print("Could not calculate speeds")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot speed over time
    plt.subplot(1, 2, 1)
    plt.plot(time_points, speeds, "b-", linewidth=2, label="Speed")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Speed (pixels/second)")
    plt.title("Object Speed Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add statistics
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

    # Plot path visualization
    plt.subplot(1, 2, 2)
    x_coords = [point[0] for point in first_moments]
    y_coords = [point[1] for point in first_moments]

    # Color points by speed (interpolate speeds for path points)
    colors = []
    colors.append(speeds[0] if speeds else 0)  # First point gets first speed
    colors.extend(speeds)  # Rest of points get their corresponding speeds

    scatter = plt.scatter(x_coords, y_coords, c=colors, cmap="viridis", s=20, alpha=0.7)
    plt.plot(x_coords, y_coords, "k-", alpha=0.3, linewidth=1)
    plt.colorbar(scatter, label="Speed (px/s)")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.title("Object Path (colored by speed)")
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    # plt.show()

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
    # plt.show()
    plt.close()

    print(f"Graph saved as: {output_file}")


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


def area_of_mask(path_to_mask):
    """
    Calculate the area of a binary mask.

    Args:
        path_to_mask (Path): Path to the binary mask image.

    Returns:
        int: Area of the mask.
    """
    mask = torch.load(path_to_mask)
    if mask is None:
        raise ValueError(f"Error: Could not read mask from {path_to_mask}")

    # Convert to float and calculate area
    mask_float = mask.float()

    area = torch.sum(mask_float)

    return area.int().item()


def first_moment_of_mask(path_to_mask):
    """
    Calculate the first moment of inertia (centroid) of a binary mask.

    Args:
        path_to_mask (Path): Path to the binary mask image.

    Returns:
        tuple: Mean x and y coordinates of the mask.
    """
    mask = torch.load(path_to_mask)
    if mask is None:
        raise ValueError(f"Error: Could not read mask from {path_to_mask}")

    mask_float = mask.float()

    area = torch.sum(mask_float)

    height, width = mask_float.shape[1:]

    x_coords = torch.arange(width).view(1, -1).expand(height, width)
    y_coords = torch.arange(height).view(-1, 1).expand(height, width)

    x = (x_coords * mask).sum() / (area + 1e-8)
    y = (y_coords * mask).sum() / (area + 1e-8)

    return x.round().int().item(), y.round().int().item()


def second_moment_of_mask(path_to_mask):
    """
    Calculate the second moment of inertia of a binary mask.

    Args:
        path_to_mask (Path): Path to the binary mask image.
    Returns:
        tuple: Second moment of inertia (Ixx, Iyy, Ixy).
    """
    mask = torch.load(path_to_mask)
    if mask is None:
        raise ValueError(f"Error: Could not read mask from {path_to_mask}")

    mask_float = mask.float()

    area = torch.sum(mask_float)

    height, width = mask_float.shape[1:]

    y_coords = torch.arange(height).view(-1, 1).expand(height, width)
    x_coords = torch.arange(width).view(1, -1).expand(height, width)

    x = (x_coords * mask).sum() / (area + 1e-8)
    y = (y_coords * mask).sum() / (area + 1e-8)

    x_diff = x_coords - x
    y_diff = y_coords - y

    xx = (x_diff**2 * mask).sum() / (area + 1e-8)
    yy = (y_diff**2 * mask).sum() / (area + 1e-8)
    xy = (x_diff * y_diff * mask).sum() / (area + 1e-8)

    return xx.int().item(), yy.int().item(), xy.int().item()


def track_object_path(video_path, output_path):
    """
    Track an object in a video and visualize its path with speed calculation.

    Args:
        video_path (Path): Path to the input video file.
        output_path (Path): Path to save the output video with tracking.
    """
    # Open video capture
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output video will be saved to: {str(output_path)}")

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return

    # Store path points and timestamps for speed calculation
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

        # Calculate area of object
        area = area_of_mask(mask_path)
        areas.append(area)

        # Calculate the first point of inertia, a.k.a. cetner of the object
        first_moment = first_moment_of_mask(mask_path)
        first_moments.append(first_moment)

        # Calculate second moment of inertia
        second_moment = second_moment_of_mask(mask_path)
        second_moments.append(second_moment)

        timestamps.append(frame_count / fps)

        # Draw the path with solid color and thickness
        if len(first_moments) > 1:
            for i in range(1, len(first_moments)):
                cv2.line(
                    frame, first_moments[i - 1], first_moments[i], (0, 255, 255), 3
                )

        # Draw smoother path
        # if len(first_moments) > 1:
        #     for i in range(1, len(first_moments), 5):
        #         if i - 5 >= 0 and i % 2 == 0:
        #             cv2.line(
        #                 frame, first_moments[i - 5], first_moments[i], (255, 255, 0), 3
        #             )

        # Draw current center point
        cv2.circle(frame, first_moment, 5, (0, 0, 255), -1)

        # Save frame if output is specified
        if output_path:
            out.write(frame)

        frame_count += 1

    # Cleanup
    cap.release()
    if output_path:
        out.release()

    # It's graph time :)
    area_args = [timestamps, areas, "Time (seconds)", "Area (pixels)", "Area Over Time"]
    create_graph(*area_args, output_file=output_path.parent / "area_graph.png")
    create_csv(*area_args[:-1], output_file=output_path.parent / "area_data.csv")

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
    create_csv(*x_args[:-1], output_file=output_path.parent / "x_data.csv")

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
    create_csv(*y_args[:-1], output_file=output_path.parent / "y_data.csv")

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
    create_csv(*xx_args[:-1], output_file=output_path.parent / "ixx_data.csv")

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
    create_csv(*yy_args[:-1], output_file=output_path.parent / "iyy_data.csv")

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
    create_csv(*xy_args[:-1], output_file=output_path.parent / "ixy_data.csv")

    create_speed_graph(
        first_moments, timestamps, output_path.parent / "speed_graph.png"
    )


def convert_mask_to_bounding_box(mask_path):
    """
    Convert a binary mask to a bounding box and save it as an image.

    Args:
        mask_path (Path): Path to the input binary mask image.
    """
    mask = read_image(mask_path)
    if mask is None:
        print(f"Error: Could not read mask from {mask_path}")
        return
    # Assumes solid mask with 128 as the background
    obj_ids = torch.unique(mask)
    print(obj_ids)
    # obj_ids.remove(128)
    # Split the color-encoded mask into a set of boolean masks
    # masks = mask == obj_ids[:, None, None]
    print(mask.size())
    print(mask)


# TODO ensure output directory exists
# TODO angle
def main():
    # convert_mask_to_bounding_box("../sam/runs/track5/mask/00000_mask.jpg")

    track_number = 8
    track = f"track{track_number}"
    Path(f"./output/{track}").mkdir(parents=True, exist_ok=True)
    track_object_path(
        Path(f"../sam/runs/{track}/video.mp4"), Path(f"./output/{track}/path.mp4")
    )
    # TODO refactor to move create_speed_graph to main
    print("Video processing complete!")


# Usage examples
if __name__ == "__main__":
    main()
