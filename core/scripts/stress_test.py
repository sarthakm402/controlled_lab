from fastapi import FastAPI
from pydantic import BaseModel
import torch
import time
from typing import List

app = FastAPI()

INPUT_SIZE = 3
MAX_BATCH_SIZE = 1000
THRESHOLD = 0.5

class StructuredInput(BaseModel):
    features: List[List[float]]

def cpu_stress(batch_len: int):
    # size = min(300 + batch_len * 5, 4000)
    size=2000
    dummy = torch.randn(size, size)
    result = torch.matmul(dummy, dummy)
    probabilities = torch.sigmoid(result.mean(dim=1)).tolist()
    probabilities = probabilities[:batch_len]
    labels = [1 if p >= THRESHOLD else 0 for p in probabilities]
    return probabilities, labels

@app.post("/predict")
def predict(req: StructuredInput):

    start = time.perf_counter()

    batch_len = len(req.features)

    if batch_len == 0:
        return {"error": "Empty batch"}

    if batch_len > MAX_BATCH_SIZE:
        return {"error": "Batch too large"}

    for row in req.features:
        if len(row) != INPUT_SIZE:
            return {"error": f"Each row must have {INPUT_SIZE} features"}

    with torch.no_grad():
        probs, labels = cpu_stress(batch_len)

    latency = time.perf_counter() - start

    return {
        "batch_size": batch_len,
        "latency_ms": round(latency * 1000, 3),
        "predictions": [
            {"probability": p, "label": l}
            for p, l in zip(probs, labels)
        ]
    }