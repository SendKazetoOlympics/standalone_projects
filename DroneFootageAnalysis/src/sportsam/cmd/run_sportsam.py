import argparse
from sportsam.io_handler import IOHandler
from sportsam.sam_handler import SAMHandler
from sportsam.analysis import Analyzer

def main():
    parser = argparse.ArgumentParser(description="Run SportsAM analysis on video files.")
    parser.add_argument("video_files", nargs="+", help="List of video files to analyze.")
    parser.add_argument("--model", type=str, default="sam2.1_hiera_base_plus", help="SAM model to use.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output frames and results.")
    
    args = parser.parse_args()

    io_handler = IOHandler()
    sam_handler = SAMHandler(model=args.model)
    analyzer = Analyzer(io_handler=io_handler, sam_handler=sam_handler, output_dir=args.output_dir)

    for video_file in args.video_files:
        analyzer.analyze_video(video_file)