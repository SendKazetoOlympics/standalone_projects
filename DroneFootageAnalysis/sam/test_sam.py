from sam2.build_sam import build_sam2_video_predictor

import os
import numpy as np
import torch

from sam_utils import (
    extract_frames_from_video,
    add_masks_to_frame,
    add_masks_to_blank,
    create_video_from_frames,
)

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

# Load the SAM2 model configuration and checkpoint
sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

# Load model
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type)


# Sam2 wants a list of JPEG images as input
video_dir = "../data/jpg_frames"
extract_frames_from_video(
    video_path="../data/sample_data.mp4",
    output_dir=video_dir,
    quality=2,
)

# Scan all the JPEG frame names
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Initialize inference state
inference_state = predictor.init_state(video_path=video_dir)

frame_idx = 0
jumper_obj_id = 1

# TODO dynamically get a clicked point
# Add a click
points = np.array([[420, 360]], dtype=np.float32)
# For labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=jumper_obj_id,
    points=points,
    labels=labels,
)

# TODO get rid of it or keep it for debugging and confirming if tracking works
# Show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx} - after adding a positive click")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
# # show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

# Run propagation throughout the video and collect the results in a dict
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# Save segments to jpg images
for frame_idx, masks_dict in video_segments.items():
    frame_path = f"../data/jpg_frames/{frame_idx:05d}.jpg"
    output_path = f"./runs/track2/video/{frame_idx:05d}_tracked.jpg"
    add_masks_to_frame(frame_path, masks_dict, output_path)
    # Add masks to a blank image as well for analysis
    blank_output_path = f"./runs/track2/blank/{frame_idx:05d}_mask.jpg"
    add_masks_to_blank((1080, 1920), masks_dict, blank_output_path)

# Combine all segments into a single video
create_video_from_frames(
    input_dir="./runs/track2/video",
    frame_format="%05d_tracked.jpg",
    output_file="./runs/track2/tracked_video.mp4",
    framerate=59.99,
)
create_video_from_frames(
    input_dir="./runs/track2/blank",
    frame_format="%05d_mask.jpg",
    output_file="./runs/track2/mask.mp4",
    framerate=59.99,
)
