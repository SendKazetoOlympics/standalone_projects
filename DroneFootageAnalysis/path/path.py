import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from pathlib import Path

import torch
from torchvision.ops import masks_to_boxes
from torchvision.io import read_image


def create_speed_graph(path_points, timestamps, output_file):
    if len(path_points) < 2:
        print("Not enough points to calculate speed")
        return

    speeds = []
    time_points = []

    # Calculate speed between consecutive points
    for i in range(1, len(path_points)):
        # Calculate distance between points (in pixels)
        x1, y1 = path_points[i - 1]
        x2, y2 = path_points[i]
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
    x_coords = [point[0] for point in path_points]
    y_coords = [point[1] for point in path_points]

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
    plt.show()

    print(f"Speed graph saved as: {output_file}")
    print(f"Average speed: {avg_speed:.2f} pixels/second")
    print(f"Maximum speed: {max_speed:.2f} pixels/second")
    print(f"Minimum speed: {min_speed:.2f} pixels/second")

    return speeds, time_points


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
        print(f"Error: Could not read mask from {path_to_mask}")
        return None

    # Convert to float and calculate area
    mask_float = mask.float()
    area = torch.sum(mask_float)

    return int(area.item())


def mean_coordinates_of_mask(path_to_mask):
    """
    Calculate the mean coordinates of a binary mask.

    Args:
        path_to_mask (Path): Path to the binary mask image.

    Returns:
        tuple: Mean x and y coordinates of the mask.
    """
    mask = torch.load(path_to_mask)
    if mask is None:
        print(f"Error: Could not read mask from {path_to_mask}")
        return None

    mask_float = mask.float()

    height, width = mask_float.shape[1:]

    x_coords = torch.arange(width)
    column_sums = mask_float[0].sum(dim=0)

    # Weighted sum of x * column_sums
    x_numerator = (x_coords * column_sums).sum()
    x_denominator = column_sums.sum()
    center_x = x_numerator / (x_denominator + 1e-8)  # Avoid division by zero

    y_coords = torch.arange(height)
    row_sums = mask_float[0].sum(dim=1)

    # Weighted sum of y * row_sums
    y_numerator = (y_coords * row_sums).sum()
    y_denominator = row_sums.sum()
    center_y = y_numerator / (y_denominator + 1e-8)  # Avoid division by zero

    return center_x.round().int().item(), center_y.round().int().item()


def track_object_path(video_path, output_path):
    """
    Track an object in a video and visualize its path with speed calculation.

    Args:
        video_path (str): Path to the input video file.
        output_path (Path): Path to save the output video with tracking.
    """
    # TODO change video_path to Path
    # Open video capture
    cap = cv2.VideoCapture(video_path)

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
    path_points = []
    timestamps = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate center point of the object
        center = mean_coordinates_of_mask(
            Path(f"../sam/runs/track7/mask_tensors/{frame_count:05d}_1_mask.pt")
        )
        if center is not None:
            path_points.append(center)
            timestamps.append(frame_count / fps)

            # Draw the path with solid color and thickness
            if len(path_points) > 1:
                for i in range(1, len(path_points)):
                    cv2.line(
                        frame, path_points[i - 1], path_points[i], (0, 255, 255), 3
                    )

            # Draw smoother path
            # if len(path_points) > 1:
            #     for i in range(1, len(path_points), 5):
            #         if i - 5 >= 0 and i % 2 == 0:
            #             cv2.line(
            #                 frame, path_points[i - 5], path_points[i], (255, 255, 0), 3
            #             )

            # Draw current center point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Display frame
        cv2.imshow("Object Tracking with Path", frame)

        # Save frame if output is specified
        if output_path:
            out.write(frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    if len(path_points) > 1:
        create_speed_graph(
            path_points, timestamps, output_path.parent / "speed_graph.png"
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

    track_number = 7
    track = f"track{track_number}"
    Path(f"./output/{track}").mkdir(parents=True, exist_ok=True)
    track_object_path(
        f"../sam/runs/{track}/video.mp4", Path(f"./output/{track}/path.mp4")
    )
    # TODO refactor to move create_speed_graph to main; for now DON'T FORGET TO CHANGE THE OUTPUT PATH
    print("Video processing complete!")


# Usage examples
if __name__ == "__main__":
    main()
