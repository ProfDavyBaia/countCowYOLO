import os
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Defina a variável ROBOFLOW_API_KEY antes de rodar.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-100").project("aerial-cows")
dataset = project.version(1).download("yolov8")

print("Dataset baixado com sucesso!")