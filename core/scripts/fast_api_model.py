from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import sys
import pandas as pd
sys.path.append("/home/sarthak/Desktop/work/ml_code/controlled_lab/core/scripts")
from model_api import MLP
import torch
import time
import logging
"""now we making a small scale robuts fast api tasks to do:
Clean Model Loading (Startup Hook)--done
Enforce Input Rules Strictly
Reject if both JSON and CSV provided.--done

Reject if neither provided.--done

Validate feature count for JSON AND CSV.--done

Reject empty batches.--done

Reject batch size > 1000 (hard cap for now).--done
Add Model Version + Configurable Thresholds--done
Structured Response Format--done
Fraud Logging (TXT for Now)--done
Latency Tracking--done
Failure Wrapper--done

Wrap inference logic in try/except.--done

If any runtime error happens:
Cap Future Scale Properly--done

If batch size > MAX_BATCH_SIZE:--done

Reject it"""

logging.basicConfig(
    filename="fraud_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()
input_size = 3
batch_size = 1000
model = None
model_version = "v1.1"
threshold = 0.5

@app.on_event("startup")
def load_model():
    global model
    model = MLP(input_size)
    model.load_state_dict(state_dict=torch.load("beset_model.pt"))
    model.eval()
    logging.info(f"Model loaded | version={model_version}")

class struct_output(BaseModel):
    features: list[list[float]]

@app.post("/predict")
def predict(req: struct_output = None, file: UploadFile = None):
    request_start_time = time.perf_counter()
    try:
        if req and file:
            logging.warning("Both JSON and CSV provided")
            return {"error": "Provide either JSON or CSV, not both."}

        if not req and not file:
            logging.warning("No input provided")
            return {"error": "No input provided."}

        if req:
            if len(req.features) == 0:
                logging.warning("Empty JSON batch")
                return {"error": "Batch cannot be empty."}

            if len(req.features) > batch_size:
                logging.warning("JSON batch size exceeded")
                return {"error": "Batch size exceeds limit."}

            for row in req.features:
                if len(row) != input_size:
                    logging.warning("Invalid JSON feature size")
                    return {"error": f"Each row must have {input_size} features."}

            x_data = torch.tensor(req.features, dtype=torch.float32)
            batch_len = len(req.features)

        else:
            df = pd.read_csv(file.file)

            if df.shape[0] == 0:
                logging.warning("Empty CSV batch")
                return {"error": "Batch cannot be empty."}

            if df.shape[0] > batch_size:
                logging.warning("CSV batch size exceeded")
                return {"error": "Batch size exceeds limit."}

            if df.shape[1] != input_size:
                logging.warning("Invalid CSV feature size")
                return {"error": f"CSV must have {input_size} features."}

            x_data = torch.tensor(df.values, dtype=torch.float32)
            batch_len = df.shape[0]

        inference_start_time = time.perf_counter()
        with torch.no_grad():
            logits = model(x_data)
            prob = torch.sigmoid(logits).view(-1).tolist()
            preds = [1 if i >= threshold else 0 for i in prob]
            predictions = [
                {"probability": p, "label": l}
                for p, l in zip(prob, preds)
            ]

        inference_latency = time.perf_counter() - inference_start_time
        request_latency = time.perf_counter() - request_start_time
        fraud_count = sum(preds)

        logging.info(
            f"Prediction complete | version={model_version} | batch={batch_len} | "
            f"fraud_count={fraud_count} | inference_latency={inference_latency:.6f}s | "
            f"request_latency={request_latency:.6f}s"
        )

        return {
            "model_version": model_version,
            "predictions": predictions,
            "threshold": threshold,
            "request_latency": request_latency,
            "inference_latency": inference_latency
        }

    except Exception as e:
        request_latency = time.perf_counter() - request_start_time
        logging.error(
            f"Inference failed | version={model_version} | error={str(e)} | "
            f"request_latency={request_latency:.6f}s"
        )
        return {
            "error": "Inference failed",
            "details": str(e),
            "request_latency": request_latency
        }