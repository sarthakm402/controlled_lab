import torch
import torch.nn as nn
torch.manual_seed(0)
x=torch.rand(200,2)
y=(x[:,0]+x[:,1]>0).float().unsqueeze(1)
lr=0.1
model=nn.Sequential(
    nn.Linear(2,5),
    nn.ReLU(),
    nn.Linear(5,1)
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for steps in range(1000):
    logits=model(x)
    loss=criterion(logits,y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if steps % 100 == 0:
        print(f"step {steps} | loss {loss.item():.4f}")