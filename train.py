from ultralytics import YOLO

# carregar modelo pré-treinado
model = YOLO("yolov8n.pt")

# iniciar treinamento
model.train(
    data="aerial-cows-1/data.yaml",
    epochs=30,
    imgsz=1024,
    batch=8
)