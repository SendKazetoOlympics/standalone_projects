## 0.2.0 (2025-07-02)

### Feat

- analyze videos in batches
- **io_hander.py**: break loaded frames into batches
- **sam_handler.py**: implement initialization of model
- **analysis.py**: refactored the moment functions
- **run_sportsam.py**: added ability to input files or manifest on cli tool
- **io_handler.py**: functionality for file manifest
- **io_handler.py**: extract files into temp directory and clean up temp directory
- part of frame extraction
- **sam_handler.py**: created init for model
- can now click multiple times for SAM 2 segmentation

### Fix

- fixed configs, type for results, added prompt for click

### Refactor

- saving before trying to hack sam2 for injecting prompt
- rethinking skeleton of how to handle data in analysis class
- changed some defaults, added more to the skeleton
- **io_handler.py**: make better use of tempfile for safety
- fix some dependencies to work better with uv
- **io_handler.py**: change if cases of extract_frames_from_video()
- fixed a bug, changed some names of files to be more accurate
- changed all paths to be dynamic instead of hardcoded (except models/checkpoints)
- ensure output paths are created and allowed for adding a point by clicking on the first frame
- changed some of the paths, still need to make dynamic
