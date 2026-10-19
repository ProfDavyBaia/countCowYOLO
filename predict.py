from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source="aerial-cows-1/test/images",
    save=True,
    imgsz=1024
)