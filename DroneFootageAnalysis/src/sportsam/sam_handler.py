"""
sam_handler.py

Handles interaction with the SAM model, including mask visualization, user input, and main segmentation logic.
"""

from sam2.build_sam import build_sam2_video_predictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import os
import enum
from jaxtyping import Int, Float


class SAMModels(enum.Enum):
    TINY = "sam2.1_hiera_tiny"
    SMALL = "sam2.1_hiera_small"
    BASE = "sam2.1_hiera_base_plus"
    LARGE = "sam2.1_hiera_large"


class SAMHandler:
    model: SAMModels

    def __init__(self, model: SAMModels | str):
        """
        Initialize the SAMHandler with a specified model.

        Args:
            model: The SAM model to use, either as a SAMModels enum or a string.
        """
        if isinstance(model, str):
            model = SAMModels(model)
        self.model = model

    def request_prompt(self, prompt: str):
        """Request a prompt from the user.

        Args:
            prompt: The prompt to display to the user.
        """
        raise NotImplementedError

    def analyze_videos(
        self, frame_direcetory: str, prompt: Float[Tensor, " n_embed"]
    ) -> list[tuple[Int, Int, Float[Tensor, "H W"]]]:
        raise NotImplementedError

    def add_new_points_or_box():
        raise NotImplementedError


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


def mouse_callback(event, x, y, _flags, _param):
    """
    Mouse callback for OpenCV to collect click coordinates.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coords = _param
        click_coords.append((x, y))
        print(f"Click at: ({x}, {y})")


# from sam_utils import (
#     extract_frames_from_video,
#     add_masks_to_frame,
#     add_masks_to_blank,
#     make_dir,
#     create_video_from_frames,
# )
def run_sam_segmentation():
    """
    Main logic for running SAM segmentation on a video.
    """

    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load the SAM2 model configuration and checkpoint
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

    # Load model
    predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint, device=device.type
    )

    # Input video path
    video_path = Path(input("Enter the path to the video file: "))

    # Extract frames
    extract_frames_from_video(video_path=video_path)

    frames_dir_path = video_path.parent / video_path.stem
    frame_names = [
        p
        for p in os.listdir(str(frames_dir_path))
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initialize inference state
    inference_state = predictor.init_state(video_path=str(frames_dir_path))

    frame_idx = 0
    jumper_obj_id = 1

    click_coords = []

    cap = cv2.VideoCapture(str(video_path))
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback, click_coords)

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read frame from video.")

    cv2.imshow("Video", frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

    labels = np.array([1] * len(click_coords), np.int32)
    points = np.array([click_coords], dtype=np.float32)
    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=jumper_obj_id,
        points=points,
        labels=labels,
    )

    # Run propagation throughout the video and collect the results in a dict
    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    output_path = make_dir(Path("./runs/track"))
    make_dir(output_path / "video")
    make_dir(output_path / "mask")
    make_dir(output_path / "mask_tensors")
    for frame_idx, masks_dict in video_segments.items():
        input_frame_path = frames_dir_path / f"{frame_idx:05d}.jpg"
        video_output_path = output_path / f"video/{frame_idx:05d}_tracked.jpg"
        add_masks_to_frame(input_frame_path, masks_dict, video_output_path)
        mask_output_path = output_path / f"mask/{frame_idx:05d}_mask.jpg"
        add_masks_to_blank((720, 1280), masks_dict, mask_output_path)

        for obj_id, mask in masks_dict.items():
            torch.save(
                torch.tensor(mask, dtype=torch.uint8),
                output_path / f"mask_tensors/{frame_idx:05d}_{obj_id}_mask.pt",
            )

    create_video_from_frames(
        input_dir=output_path / "video",
        frame_format="%05d_tracked.jpg",
        output_file=output_path / "video.mp4",
        framerate=56,
    )
    create_video_from_frames(
        input_dir=output_path / "mask",
        frame_format="%05d_mask.jpg",
        output_file=output_path / "mask.mp4",
        framerate=56,
    )
