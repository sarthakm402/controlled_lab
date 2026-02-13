import torch
import torch.nn as nn
torch.manual_seed(1)
x=torch.randn(200,1)
y=(x>0).float
model=nn.Linear(1,1)
criterion= nn.BCEWithLogitsLoss()
optimiser=torch.optim.SGD(model.parameters(), lr=0.5)
for steps in range(1000):
    pred=model(x)
    loss=criterion(pred,y)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    if steps % 100 == 0:
        print(
            f"step {steps} | loss {loss.item():.4f} | "
            f"W {model.weight.item():.4f}"
        )