import torch
x=torch.tensor([1,2,3,4],requires_grad=True)
y=torch.sum(x)
loss=y**2
loss.backward()
print("x grad is",x.grad)
lr=0.1
with torch.no_grad():
    x=x-lr*x.grad
x.grad.zero_()
