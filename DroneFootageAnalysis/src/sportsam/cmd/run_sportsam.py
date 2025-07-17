import argparse
import tempfile

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

    with tempfile.TemporaryDirectory() as temp_dir:
        io_handler = IOHandler(temp_dir=temp_dir, output_dir=args.output_dir)

        if args.video_files:
            io_handler.extract_frames_from_videos(args.video_files)
        elif args.manifest:
            io_handler.extract_frames_from_manifest(args.manifest)

        sam_handler = SAMHandler(frames_path=temp_dir, model=args.model)
        sam_handler.request_click()
        results = sam_handler.segment_videos(io_handler.temp_dir)

        io_handler.save_output_masks(results)
        io_handler.group_frames_by_video()
        # Unbatch the temp files for use with write_centroid()
        io_handler.unbatch_frames()

        video_dirs = [p for p in io_handler.output_dir.iterdir() if p.is_dir()]
        video_dirs = sorted(video_dirs, key=lambda p: p.name)
        for video_idx, video_dir in enumerate(video_dirs):
            zeroth_moments = Analyzer.zeroth_image_moment(video_dir)
            first_moments = Analyzer.first_image_moment(video_dir)
            second_moments = Analyzer.second_image_moment(video_dir)

            # TODO turn moments data into lists?
            # IOHandler.create_graph(video_dir, "Area Over Time", zeroth_moments)
            # create all the graphs and csvs
            # io_handler.create_graph()
            io_handler.write_centroid(first_moments, video_idx)
            # io_handler.recreate_video_from_frames()
