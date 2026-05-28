from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/Engagement_level.v2i.yolov8/data.yaml",
    epochs=5,
    imgsz=320,
    batch=4
)