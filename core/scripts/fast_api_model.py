from fastapi import FastAPI,File,UploadFile,Form
from pydantic import BaseModel
import sys
import pandas as pd
sys.path.append("/home/sarthak/Desktop/work/ml_code/controlled_lab/core/scripts")
from model_api import MLP
import torch
app=FastAPI()
input_size=3
model=MLP(input_size)
model.load_state_dict(state_dict=torch.load("beset_model.pt"))
model.eval()
class struct_output(BaseModel):
    features:list[list[float]]
@app.post("/predict")
def predict(req:struct_output=None,file:UploadFile=None):
    if req:
     x_data = torch.tensor(req.features,dtype=torch.float32)
    elif file:
       df=pd.read_csv(file.file)
       if df.shape[1] != input_size:
            return {"error": f"CSV must have {input_size} features"}
       x_data = torch.tensor(df.values, dtype=torch.float32)  
        
    else:
        return {"error": "No input provided"}
    with torch.no_grad():
        logits = model(x_data)
        prob = torch.sigmoid(logits).squeeze().tolist()
    return {"probability": prob}
