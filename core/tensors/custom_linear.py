"""linear model"""
import torch
torch.manual_seed(0)
class mylinear():
    def __init__(self,in_features,out_features):
        self.w=torch.randn(in_features,out_features,requires_grad=True)
        self.b=torch.zeros(out_features,requires_grad=True)
    def forward_pass(self,x):
        return x@self.w+self.b
    @property
    def parameters(self):
        return [self.W, self.b]
x= torch.randn(200, 2)
y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
model=mylinear(2,1)
lr=0.1
for steps in range(1000):
    logits=model(x)
    probs=torch.sigmoid(logits)
    loss=-(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters:
            param -= lr * param.grad
            param.grad.zero_()
    if steps % 100 == 0:
        print(f"step {steps} | loss {loss.item():.4f} | W {model.w[0,0].item():.4f}")
"""mlp model"""
class mlp():
    def __init__(self):
        self.l1=mylinear(2,5)
        self.l2=mylinear(5,1)
    def forward_pass(self,x):
        self.z1=self.l1.forward_pass(x)
        self.a1=torch.relu(self.z1)
        self.z2=self.l2.forward_pass(self.a1)
        return self.z2
    @property
    def parameters(self):
        return self.l1.parameters + self.l2.parameters


