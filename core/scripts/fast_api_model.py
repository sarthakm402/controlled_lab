from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import sys
import pandas as pd
import torch
import time
import logging
from typing import Optional, List

sys.path.append("/home/sarthak/Desktop/work/ml_code/controlled_lab/core/scripts")
from model_api import MLP
logging.basicConfig(
    filename="fraud_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

INPUT_SIZE = 3
MAX_BATCH_SIZE = 1000
MODEL_VERSION = "v1.1"
THRESHOLD = 0.5

model = None
def build_response(
    status: str,
    message: str,
    batch_size: int,
    latency_seconds: float,
    data: Optional[dict] = None
):
    return {
        "status": status, 
        "message": message,
        "model_version": MODEL_VERSION,
        "batch_size": batch_size,
        "latency_ms": round(latency_seconds * 1000, 3),
        "data": data
    }

@app.on_event("startup")
def load_model():
    global model
    model = MLP(INPUT_SIZE)
    model.load_state_dict(torch.load("beset_model.pt"))
    model.eval()
    logging.info(f"Model loaded | version={MODEL_VERSION}")

class StructuredInput(BaseModel):
    features: List[List[float]]


@app.post("/predict")
def predict(req: Optional[StructuredInput] = None, file: Optional[UploadFile] = None):
    request_start_time = time.perf_counter()

    try:
      
        if req and file:
            latency = time.perf_counter() - request_start_time
            return build_response(
                "error",
                "Provide either JSON or CSV, not both.",
                0,
                latency
            )

        if not req and not file:
            latency = time.perf_counter() - request_start_time
            return build_response(
                "error",
                "No input provided.",
                0,
                latency
            )

        if req:
            batch_len = len(req.features)

            if batch_len == 0:
                latency = time.perf_counter() - request_start_time
                return build_response(
                    "error",
                    "Batch cannot be empty.",
                    0,
                    latency
                )

            if batch_len > MAX_BATCH_SIZE:
                latency = time.perf_counter() - request_start_time
                return build_response(
                    "error",
                    "Batch size exceeds limit.",
                    batch_len,
                    latency
                )

            for row in req.features:
                if len(row) != INPUT_SIZE:
                    latency = time.perf_counter() - request_start_time
                    return build_response(
                        "error",
                        f"Each row must have {INPUT_SIZE} features.",
                        batch_len,
                        latency
                    )

            x_data = torch.tensor(req.features, dtype=torch.float32)

        else:
            df = pd.read_csv(file.file)
            batch_len = df.shape[0]

            if batch_len == 0:
                latency = time.perf_counter() - request_start_time
                return build_response(
                    "error",
                    "Batch cannot be empty.",
                    0,
                    latency
                )

            if batch_len > MAX_BATCH_SIZE:
                latency = time.perf_counter() - request_start_time
                return build_response(
                    "error",
                    "Batch size exceeds limit.",
                    batch_len,
                    latency
                )

            if df.shape[1] != INPUT_SIZE:
                latency = time.perf_counter() - request_start_time
                return build_response(
                    "error",
                    f"CSV must have {INPUT_SIZE} features.",
                    batch_len,
                    latency
                )

            x_data = torch.tensor(df.values, dtype=torch.float32)

        inference_start = time.perf_counter()

        with torch.no_grad():
            logits = model(x_data)
            probabilities = torch.sigmoid(logits).view(-1).tolist()
            labels = [1 if p >= THRESHOLD else 0 for p in probabilities]

        inference_latency = time.perf_counter() - inference_start
        request_latency = time.perf_counter() - request_start_time

        predictions = [
            {"probability": p, "label": l}
            for p, l in zip(probabilities, labels)
        ]

        fraud_count = sum(labels)

        logging.info(
            f"Prediction complete | version={MODEL_VERSION} | "
            f"batch={batch_len} | fraud_count={fraud_count} | "
            f"inference_latency={inference_latency:.6f}s | "
            f"request_latency={request_latency:.6f}s"
        )

        return build_response(
            "success",
            "Inference completed successfully.",
            batch_len,
            request_latency,
            data={
                "predictions": predictions,
                "threshold": THRESHOLD,
                "fraud_count": fraud_count,
                "inference_latency_ms": round(inference_latency * 1000, 3)
            }
        )

    except Exception as e:
        request_latency = time.perf_counter() - request_start_time

        logging.error(
            f"Inference failed | version={MODEL_VERSION} | "
            f"error={str(e)} | request_latency={request_latency:.6f}s"
        )

        return build_response(
            "error",
            "Inference failed due to internal error.",
            0,
            request_latency
        )