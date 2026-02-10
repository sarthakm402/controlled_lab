import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.randn(100, 1)
y = 3.0 * X + 2.0 + 0.1 * torch.randn(100, 1)
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range (1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(
            f"step {step} | loss {loss.item():.4f} | "
            f"W {model.weight.item():.4f}"
        )