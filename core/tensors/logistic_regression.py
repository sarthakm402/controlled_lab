import torch
import time 
torch.manual_seed(1)
x=torch.rand(200,1)
y = (x> 0).float()
w=torch.zeros(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)
lr=0.5
start_time=time.time()
for step in range(1000):
    logits = x * w + b
    probs = torch.sigmoid(logits)
    loss=-(y * torch.log(probs + 1e-8) +
             (1 - y) * torch.log(1 - probs + 1e-8)).mean()
    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if step % 100 == 0:
        print(
            f"step {step} | loss {loss.item():.4f} | "
            f"W {w.item():.4f}"
        )
end_time=time.time()
print(f"Total training time: {end_time - start_time:.6f} seconds")
"""when w is random initialised and bias to 0 speed:1.792961
when w is random and b is random :1.794776
when w is 0 and b is 0 :1.778072"""
