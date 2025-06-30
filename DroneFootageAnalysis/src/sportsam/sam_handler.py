"""
sam_handler.py

Handles interaction with the SAM model, including mask visualization, user input, and main segmentation logic.
"""

import enum
import urllib.request
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Int, Float
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames


class SAMModels(enum.Enum):
    TINY = "sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"
    SMALL = "sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"
    BASE_PLUS = "sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"
    LARGE = "sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"


class SAMHandler:
    model: SAMModels
    predictor: SAM2VideoPredictor
    inference_state: dict[str, Any]
    frames_path: Path

    # TODO add support for loading inference state
    def __init__(self, frames_path: Path | str, model: SAMModels | str):
        """Initialize the SAMHandler with the specified frames directory and model.

        Args:
            frames_path (Path | str): Path to the directory containing extracted video frames.
            model (SAMModels | str ): The SAM model variant to use. Can be a SAMModels enum or a string matching one of its values.

        Raises:
            ValueError: If the provided model string does not correspond to a valid SAMModels enum member.
        """
        if isinstance(frames_path, str):
            frames_path = Path(frames_path)
        self.frames_path = frames_path
        # Accept both enum names (e.g. "SMALL") and values (e.g. "sam2.1_hiera_s.yaml")
        if isinstance(model, str):
            try:
                model = SAMModels[model]
            except KeyError:
                try:
                    model = SAMModels(model)
                except ValueError:
                    raise ValueError(
                        f"{model!r} is not a valid SAMModels name or value"
                    )
        self.model = model
        print(f"Using model: {model}")

        # Use system default config folder
        config_base = Path.home() / ".config" / "sportsam"
        config_base.mkdir(parents=True, exist_ok=True)
        config_path = config_base / self.model.value[0]
        checkpoint_path = config_base / self.model.value[1]

        # Updated base URLs
        config_base_url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/"
        checkpoint_base_url = (
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
        )

        config_url = f"{config_base_url}{self.model.value[0]}"
        checkpoint_url = f"{checkpoint_base_url}{self.model.value[1]}"

        if not config_path.exists():
            print(f"Downloading {self.model} model config to {config_path}...")
            urllib.request.urlretrieve(config_url, config_path)
            print("Download complete.")
        else:
            print(f"{self.model} model config found.")

        if not checkpoint_path.exists():
            print(f"Downloading {self.model} model checkpoint to {checkpoint_path}...")
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
            print("Download complete.")
        else:
            print(f"{self.model} model checkpoints found.")

        # Select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might give numerically different outputs and sometimes degraded performance on MPS. See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        print(f"Using device: {device}")

        self.predictor = build_sam2_video_predictor(
            config_path, checkpoint_path, device=device.type
        )

        self.inference_state = self.predictor.init_state(
            video_path=str(self.frames_path / "batch0")
        )

    def analyze_videos(
        self, frames_dir: Path
    ) -> list[tuple[Int, Int, Float[torch.Tensor, "H W"]]]:
        """Analyze all batches of frames in a directory.

        Args:
            frames_dir: Directory containing batch subdirectories of video frames.

        Returns:
            List of tuples (frame_idx, obj_ids, mask_tensor)
        """
        results = []

        # Find all batch directories (batch0, batch1, ...)
        batch_dirs = sorted(
            [
                d
                for d in frames_dir.iterdir()
                if d.is_dir() and d.name.startswith("batch")
            ],
            key=lambda d: int(d.name.replace("batch", "")),
        )

        for batch_dir in batch_dirs:
            images, _, _ = load_video_frames(
                video_path=batch_dir,
                image_size=self.predictor.image_size,
                offload_video_to_cpu=False,
                compute_device=self.predictor.device,
            )
            self.inference_state["images"] = images
            self.inference_state["num_frames"] = len(images)

            for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=0
            ):
                results.append((frame_idx, obj_ids, masks))

        # TOOD save final inference state?

        return results


##### TODO refactor everything below #####


# def show_mask(mask, ax, obj_id=None, random_color=False):
#     """Show a mask on a matplotlib axis with a specific color.

#     Args:
#         mask: A binary mask of shape (H, W) or (1, H, W).
#         ax: The matplotlib axis to draw on.
#         obj_id: Optional object ID for color mapping.
#         random_color: If True, use a random color instead of a fixed one.

#     """
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         cmap = plt.get_cmap("tab10")
#         cmap_idx = 0 if obj_id is None else obj_id
#         color = np.array([*cmap(cmap_idx)[:3], 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


# def mouse_callback(event, x, y, _flags, _param):
#     """
#     Mouse callback for OpenCV to collect click coordinates.
#     """
#     if event == cv2.EVENT_LBUTTONDOWN:
#         click_coords = _param
#         click_coords.append((x, y))
#         print(f"Click at: ({x}, {y})")


# # from sam_utils import (
# #     extract_frames_from_video,
# #     add_masks_to_frame,
# #     add_masks_to_blank,
# #     make_dir,
# #     create_video_from_frames,
# # )
# def run_sam_segmentation():
#     """
#     Main logic for running SAM segmentation on a video.
#     """

#     # Select the device for computation
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")
#     print(f"using device: {device}")

#     if device.type == "cuda":
#         torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#         if torch.cuda.get_device_properties(0).major >= 8:
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.backends.cudnn.allow_tf32 = True

#     # Load the SAM2 model configuration and checkpoint
#     sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
#     model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

#     # Load model
#     predictor = build_sam2_video_predictor(
#         model_cfg, sam2_checkpoint, device=device.type
#     )

#     # Input video path
#     video_path = Path(input("Enter the path to the video file: "))

#     # Extract frames
#     extract_frames_from_video(video_path=video_path)

#     frames_dir_path = video_path.parent / video_path.stem
#     frame_names = [
#         p
#         for p in os.listdir(str(frames_dir_path))
#         if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
#     ]
#     frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#     # Initialize inference state
#     inference_state = predictor.init_state(video_path=str(frames_dir_path))

#     frame_idx = 0
#     jumper_obj_id = 1

#     click_coords = []

#     cap = cv2.VideoCapture(str(video_path))
#     cv2.namedWindow("Video")
#     cv2.setMouseCallback("Video", mouse_callback, click_coords)

#     ret, frame = cap.read()
#     if not ret:
#         raise ValueError("Could not read frame from video.")

#     cv2.imshow("Video", frame)
#     cv2.waitKey(0)
#     cap.release()
#     cv2.destroyAllWindows()

#     labels = np.array([1] * len(click_coords), np.int32)
#     points = np.array([click_coords], dtype=np.float32)
#     _, _, _ = predictor.add_new_points_or_box(
#         inference_state=inference_state,
#         frame_idx=frame_idx,
#         obj_id=jumper_obj_id,
#         points=points,
#         labels=labels,
#     )

#     # Run propagation throughout the video and collect the results in a dict
#     video_segments = {}
#     for (
#         out_frame_idx,
#         out_obj_ids,
#         out_mask_logits,
#     ) in predictor.propagate_in_video(inference_state):
#         video_segments[out_frame_idx] = {
#             out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#             for i, out_obj_id in enumerate(out_obj_ids)
#         }

#     output_path = make_dir(Path("./runs/track"))
#     make_dir(output_path / "video")
#     make_dir(output_path / "mask")
#     make_dir(output_path / "mask_tensors")
#     for frame_idx, masks_dict in video_segments.items():
#         input_frame_path = frames_dir_path / f"{frame_idx:05d}.jpg"
#         video_output_path = output_path / f"video/{frame_idx:05d}_tracked.jpg"
#         add_masks_to_frame(input_frame_path, masks_dict, video_output_path)
#         mask_output_path = output_path / f"mask/{frame_idx:05d}_mask.jpg"
#         add_masks_to_blank((720, 1280), masks_dict, mask_output_path)

#         for obj_id, mask in masks_dict.items():
#             torch.save(
#                 torch.tensor(mask, dtype=torch.uint8),
#                 output_path / f"mask_tensors/{frame_idx:05d}_{obj_id}_mask.pt",
#             )

#     create_video_from_frames(
#         input_dir=output_path / "video",
#         frame_format="%05d_tracked.jpg",
#         output_file=output_path / "video.mp4",
#         framerate=56,
#     )
#     create_video_from_frames(
#         input_dir=output_path / "mask",
#         frame_format="%05d_mask.jpg",
#         output_file=output_path / "mask.mp4",
#         framerate=56,
#     )
