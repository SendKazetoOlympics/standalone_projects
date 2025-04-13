# Drone footage analyzer

## Background

High jump is a speed conversion event, in which the athlete gathers horizontal momentum during the approach and converts that into vertical momentum that ultimately results in clearance in height.
This makes the approach the most important part of high jump.
Aerial footages are excellent sources of data to extract useful analytics and help athletes improve.
However, just looking at the videos and relying on human insight is not a very reliable way to make systematic improvement. 
In this project, you will work on developing a pipeline to extract meaning analytics from aerial footages of the high jump event.

## Tasks

You will be given a dataset mainly consist of videos of top-down view of a jump.
This angle provides very clear signal for extracting the curve without much of the need to worry about perspectives.
Given a video, these are the 

1. Fit the curve of the approach 
2. Estimate the subject's velocity in pixel coordinate
3. Estimate the stride length of each steps, and potentially the tempo.

## References

1. [OpenCV](https://opencv.org/)
2. [Ultralytics](https://www.ultralytics.com/)
3. [Computer vision: Algorithms and Applications](https://szeliski.org/Book/)