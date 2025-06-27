import argparse
from sportsam.io_handler import IOHandler
from sportsam.sam_handler import SAMHandler
from sportsam.analysis import Analyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run SportSAM analysis on video files."
    )
    parser.add_argument(
        "video_files", nargs="+", help="List of video files to analyze."
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

    io_handler.extract_frames_from_videos(args.video_files)

