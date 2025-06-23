import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


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
        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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


def track_object_path(video_path, output_path=None):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")

    # Initialize tracker (you can use different trackers)
    tracker = cv2.TrackerCSRT.create()

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return

    # Select ROI (Region of Interest) - the object to track
    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")

    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    # Store path points and timestamps for speed calculation
    path_points = []
    timestamps = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            # Draw bounding box
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate center point of the object
            center = (int(x + w / 2), int(y + h / 2))
            path_points.append(center)
            timestamps.append(frame_count / fps)

            # Draw the path with solid color and thickness
            if len(path_points) > 1:
                for i in range(1, len(path_points)):
                    cv2.line(
                        frame, path_points[i - 1], path_points[i], (0, 255, 255), 3
                    )

            # Draw current center point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        else:
            # If tracking fails, try to reinitialize or handle the failure
            cv2.putText(
                frame,
                "Tracking lost",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Display frame
        cv2.imshow("Object Tracking with Path", frame)

        # Save frame if output is specified
        if output_path:
            out.write(frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    if len(path_points) > 1:
        create_speed_graph(path_points, timestamps, "./output/track3/speed_graph.png")


def main():
    track_object_path("../sam/runs/track3/mask.mp4", "./output/track3/path.mp4")
    print("Video processing complete! Check output file.")


# Usage examples
if __name__ == "__main__":
    main()
