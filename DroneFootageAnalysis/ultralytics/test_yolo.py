from ultralytics import YOLO

# Medium model
model = YOLO("yolo11m.pt")
model.track(
    source="data/sample_data.mp4",
    show=True,
    save=True,
    project="./runs",
    imgsz=(1920, 1080),
    max_det=1,
    classes=[0],
    conf=0.1,
)
