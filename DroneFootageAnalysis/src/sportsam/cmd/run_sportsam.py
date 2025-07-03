import argparse
import tempfile

import hydra

from sportsam.io_handler import IOHandler
from sportsam.sam_handler import SAMHandler
from sportsam.analysis import Analyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run SportSAM analysis on video files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="List of video files to analyze.",
        dest="video_files",
    )
    group.add_argument(
        "-m",
        "--manifest",
        type=str,
        default="manifest.csv",
        help="Path to the manifest file containing list of videos.",
        dest="manifest",
    )
    parser.add_argument("--model", type=str, default="small", help="SAM model to use.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output frames and results.",
    )
    parser.add_argument(
        "--save_inference_state",
        action=argparse.BooleanOptionalAction,
        help="Save the inference state after processing.",
    )
    # TODO custom checkpoint path?
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        #    Does this work?
        # default=~/.config/sportsam/checkpoints,
        help="Path to the directory where your checkpoints are stored.",
    )
    # TODO click vs input inference state

    args = parser.parse_args()

    # TODO rename temp_dir (in io_handler) as frames_dir so we can make frame directory permanent?
    with tempfile.TemporaryDirectory() as temp_dir:
        io_handler = IOHandler(temp_dir=temp_dir, output_dir=args.output_dir)

        if args.video_files:
            io_handler.extract_frames_from_videos(args.video_files)
        elif args.manifest:
            io_handler.extract_frames_from_manifest(args.manifest)

        sam_handler = SAMHandler(frames_path=temp_dir, model=args.model)
        sam_handler.request_click()
        results = sam_handler.analyze_videos(io_handler.temp_dir)
        io_handler.save_output_masks(results)

        # analyzer = Analyzer()
        # sam_handler.analyze_videos(temp_dir)
