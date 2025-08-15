# DroneFootageAnalysis

A standalone utility for analyzing drone footage in sports contexts.

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

