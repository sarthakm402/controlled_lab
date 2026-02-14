import torch
torch.manual_seed(0)
x=torch.rand(200,2)
y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1) 
input_dim = 2
hidden_dim = 5
output_dim = 1
w1=torch.randn(input_dim,hidden_dim,requires_grad=True)
b1=torch.zeros(hidden_dim,requires_grad=True)
w2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
b2 = torch.zeros(output_dim, requires_grad=True)
lr = 0.1
for steps in range(1000):
    z1=x@w1+b1
    a1=torch.relu(z1)
    z2=a1@w2+b2
    probs=torch.sigmoid(z2)
    loss=-(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
    loss.backward()
    with torch.no_grad():
        for param in [w1, b1, w2, b2]:
            param -= lr * param.grad
            param.grad.zero_()
    if steps % 100 == 0:
        print(f"step {steps} | loss {loss.item():.4f}")