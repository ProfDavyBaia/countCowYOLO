import pandas as pd

file_path = "runs/detect/train/results.csv"

df = pd.read_csv(file_path)
last = df.iloc[-1]

print("\n=== MÉTRICAS FINAIS ===\n")
print(f"Precision: {last['metrics/precision(B)']:.4f}")
print(f"Recall: {last['metrics/recall(B)']:.4f}")
print(f"mAP@0.5: {last['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {last['metrics/mAP50-95(B)']:.4f}")