# Tempo extractor

## Background

One of the key elemenets in the high jump approach is to have a stable, smooth, and quickening approach.
Given the large amount of videos athletes collect over their training period, it would be ideal if there is a tool that can automatically extract the ground contact time of every foot strike as well as the time between one ground to another.
In this project, you will create an active learning software loop which you will first annotate some images manually, then train a machine learning algorithm to annotate videos automatically.

## Task

1. Annotate a small set of videos.
2. Train a YOLO + time coherence model to identify periods of interest
3. Experiment with better vision and time coherence model

## Reference

1. [YOLO](https://docs.ultralytics.com/)
2. [Label studio](https://labelstud.io/guide/ml)