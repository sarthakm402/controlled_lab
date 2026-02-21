import torch 
import torch.nn as nn
from fastapi import FastAPI
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("my csv path")
x=df.drop(columns=["id","label"])
y=df["label"]
xtrain,xtemp,ytrain,ytemp=train_test_split(x,y,test_size=0.3,shuffle=True)
xval,xtest,yval,ytest=train_test_split(xtemp,ytemp,test_size=0.5,shuffle=True)
scaler=StandardScaler()
xtrain_scaled=scaler.fit_transform(xtrain)
xval_scaled=scaler.transform(xval)
xtest_scaled=scaler.transform(xtest)

class mydataset(Dataset):
    def __init__(self,x,y):
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
train_loader=DataLoader(mydataset(xtrain_scaled,ytrain),
                        batch_size=254,
                        shuffle=True)
val_loader=DataLoader(mydataset(xval_scaled,yval),batch_size=254)
test_loader=DataLoader(mydataset(xtest_scaled,ytest),batch_size=254)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
class MLP(nn.Module):
    def __init__(self,input_features):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_features,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.net(x).squeeze(1)
model=MLP(xtrain_scaled.shape[1]).to(device)
num_pos = (ytrain == 1).sum()
num_neg = (ytrain == 0).sum()
pos_weight_value = num_neg / (num_pos+1e-8)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
best_val_loss = float("inf")
patience = 10
counter = 0
criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser=torch.optim.Adam(model.parameters(),lr=0.0001)
for steps in range(1000):
    model.train()
    train_loss=[]
    for xb,yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits=model(xb)
        loss=criterion(logits,yb)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        train_loss.append(loss.item())
    model.eval()
    val_loss=[]
    with torch.no_grad():
        for xb1,yb1 in val_loader:
         xb1,yb1=xb1.to(device),yb1.to(device)
         logits=model(xb1)
         loss=criterion(logits,yb1)
         val_loss.append(loss.item())
    avg_val_loss = np.mean(val_loss)

    if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
    else:
          counter += 1

    if counter >= patience:
           print("Early stopping triggered")
           break

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
     for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)  
        all_preds.extend(probs.cpu().numpy().flatten())
        all_targets.extend(yb.cpu().numpy())
preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
acc = accuracy_score(all_targets, preds_binary)
prec = precision_score(all_targets, preds_binary)
rec = recall_score(all_targets, preds_binary)
roc_auc = roc_auc_score(all_targets, all_preds)
pr_auc = average_precision_score(all_targets, all_preds)
print(f"Test Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | ROC-AUC: {roc_auc:.4f}| PR-AUC:{pr_auc:.4f}")
torch.save(model.state_dict(), "best_model.pt")


    
         





    

