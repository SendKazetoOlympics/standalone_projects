![status: WIP](https://img.shields.io/badge/status-WIP-orange?style=flat-square)

# Status: Work in Progress

This project is still a **work in progress**.  
The codebase is actively changing, and many parts are incomplete, experimental, or fragile.

- Expect breaking changes and incomplete functionality.
- Known issues include memory leaks, OpenCV quirks, and environment-specific bugs.
- Use at your own risk; stability is not guaranteed.

This repository should be considered a **research prototype** rather than production-ready software.

# DroneFootageAnalysis

A standalone utility for analyzing drone footage in sports contexts. This CLI tool is structured around a modular pipeline for analyzing sports videos using SAM2-based segmentation and OpenCV-driven analysis. The program is divided into distinct components:

- I/O Handling (`io_handler`) – Responsible for loading and saving videos, masks, prompts, and results. It manages per-video frame grouping, caching, and export to CSV or visualized output.
- Segmentation Engine (`sam_handler`) – Wraps the SAM2 model, providing functions for frame propagation, mask extraction, and prompt-based segmentation. It supports batching, caching of inference state, and prompt reuse across videos.
- Analysis Suite (`analysis`) – Computes quantitative metrics over time, including mask area, center of mass, velocity, trajectory curve, and second moment of inertia. Results are stored in structured dictionaries for flexible plotting and downstream analytics.

The overall workflow follows a prompt → segmentation → propagation → analysis cycle. Each video is processed independently, with results serialized to both visual artifacts (videos/frames) and numerical outputs (CSV). This modular design allows scaling from single-video experiments to larger multi-video studies while isolating heavy memory operations within the segmentation engine.

## Installation

Install directly from the repository:

```bash
pip install git+https://github.com/SendKazetoOlympics/standalone_projects/@drone_dev#subdirectory=DroneFootageAnalysis
```

## Usage

```bash
sportsam -f <file_path>
```

- The current implementation does not account for pre-existing output folders.
- It is recommended to clear the output directory before running the tool to avoid conflicts:

```bash
rm -rf output/
```

## Colab

- A Colab notebook is available for quick testing and demonstration of the tool's capabilities. You can access it [here](https://colab.research.google.com/drive/1p9vopbo_L6B5Qqu1DA1luHLx88_sgmO9?usp=sharing).
