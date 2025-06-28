import argparse
import tempfile
from pathlib import Path

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
    parser.add_argument(
        "--model", type=str, default="sam2.1_hiera_small", help="SAM model to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output frames and results.",
    )

    args = parser.parse_args()

    io_handler = IOHandler()
    # sam_handler = SAMHandler(model=args.model)
    # analyzer = Analyzer()

    with tempfile.TemporaryDirectory() as temp_dir:
        if args.video_files:
            io_handler.extract_frames_from_videos(args.video_files, Path(temp_dir))
        elif args.manifest:
            io_handler.extract_frames_from_manifest(args.manifest, Path(temp_dir))
