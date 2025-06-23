from sam2.build_sam import build_sam2_video_predictor

import os
import numpy as np
import torch
from pathlib import Path

from sam_utils import *

# TODO right now all the paths are hardcoded, make them configurable, for now run them from within the `sam` dirtory

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
    # Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# TODO try tiny model
# Load the SAM2 model configuration and checkpoint
sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

# Load model
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type)

# Input video path
video_path = Path(input("Enter the path to the video file: "))

# Sam2 wants a list of JPEG images as input
extract_frames_from_video(
    video_path=video_path,
)

frames_dir_path = video_path.parent / video_path.stem
# Scan all the JPEG frame names
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
pit_id = 2

# TODO dynamically get a clicked point
# For labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

# Add a click for the jumper
points = np.array([[420, 360]], dtype=np.float32)
_, _, _ = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=jumper_obj_id,
    points=points,
    labels=labels,
)

# Add a click for the pit
points = np.array([[1080, 500]], dtype=np.float32)
_, _, _ = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=pit_id,
    points=points,
    labels=labels,
)


# Run propagation throughout the video and collect the results in a dict
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# TODO make paths Path objects
# TODO Ensure output directories exist
# Save segments to jpg images
for frame_idx, masks_dict in video_segments.items():
    frame_path = frames_dir_path / f"{frame_idx:05d}.jpg"
    output_path = f"./runs/track3/video/{frame_idx:05d}_tracked.jpg"
    add_masks_to_frame(frame_path, masks_dict, output_path)
    blank_output_path = f"./runs/track3/blank/{frame_idx:05d}_mask.jpg"
    add_masks_to_blank((1080, 1920), masks_dict, blank_output_path)

# Combine all segments into a single video
create_video_from_frames(
    input_dir="./runs/track4/video",
    frame_format="%05d_tracked.jpg",
    output_file="./runs/track4/tracked_video.mp4",
    framerate=59.99,
)
create_video_from_frames(
    input_dir="./runs/track4/blank",
    frame_format="%05d_mask.jpg",
    output_file="./runs/track4/mask.mp4",
    framerate=59.99,
)
