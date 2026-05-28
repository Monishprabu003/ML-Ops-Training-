from ultralytics import YOLO

model = YOLO("runs/detect/train-3/weights/best.pt")

results = model.predict(
    source="test.jpg",
    show=True
)

print("Prediction Completed")