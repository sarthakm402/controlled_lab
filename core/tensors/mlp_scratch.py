"""using relu activation"""
# import torch
# torch.manual_seed(0)
# x=torch.rand(200,2)
# y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1) 
# input_dim = 2
# hidden_dim = 5
# output_dim = 1
# w1=torch.randn(input_dim,hidden_dim,requires_grad=True)
# b1=torch.zeros(hidden_dim,requires_grad=True)
# w2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
# b2 = torch.zeros(output_dim, requires_grad=True)
# lr = 0.1
# for steps in range(1000):
#     z1=x@w1+b1
#     a1=torch.relu(z1)
#     z2=a1@w2+b2
#     probs=torch.sigmoid(z2)
#     loss=-(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
#     loss.backward()
#     with torch.no_grad():
#         for param in [w1, b1, w2, b2]:
#             param -= lr * param.grad
#             param.grad.zero_()
#     if steps % 100 == 0:
#         print(f"step {steps} | loss {loss.item():.4f}")

"""using sigmoid activation vanisihing gradient"""
# import torch
# torch.manual_seed(0)
# x=torch.rand(200,2)
# y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1) 
# input_dim = 2
# hidden_dim = 5
# output_dim = 1
# w1=torch.randn(input_dim,hidden_dim,requires_grad=True)
# b1=torch.zeros(hidden_dim,requires_grad=True)
# w2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
# b2 = torch.zeros(output_dim, requires_grad=True)
# lr = 0.1
# for steps in range(1000):
#     z1=x@w1+b1
#     a1=torch.sigmoid(z1)
#     z2=a1@w2+b2
#     probs=torch.sigmoid(z2)
#     loss=-(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
#     loss.backward()
#     with torch.no_grad():
#         for param in [w1, b1, w2, b2]:
#             param -= lr * param.grad
#             param.grad.zero_()
#     if steps % 100 == 0:
#      print(f"step {steps}")
#      print("loss:", loss.item())
#      print("W1 grad norm:", w1.grad.norm().item())
#      print("W2 grad norm:", w2.grad.norm().item())
#      print("-----")
"""exploding graidents"""
# import torch
# torch.manual_seed(0)
# x=torch.rand(200,2)
# y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1) 
# input_dim = 2
# hidden_dim = 5
# output_dim = 1
# w1 = torch.randn(input_dim, hidden_dim) * 5
# w1.requires_grad_()
# w2 = torch.randn(hidden_dim, output_dim) * 5
# w2.requires_grad_()
# b1=torch.zeros(hidden_dim,requires_grad=True)
# b2 = torch.zeros(output_dim, requires_grad=True)
# lr = 0.1
# for steps in range(1000):
#     z1=x@w1+b1
#     a1=torch.sigmoid(z1)
#     z2=a1@w2+b2
#     probs=torch.sigmoid(z2)
#     loss=-(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
#     loss.backward()
#     with torch.no_grad():
#         for param in [w1, b1, w2, b2]:
#             param -= lr * param.grad
#             param.grad.zero_()
#     if steps % 100 == 0:
#      print(f"step {steps}")
#      print("loss:", loss.item())
#      print("W1 grad norm:", w1.grad.norm().item())
#      print("W2 grad norm:", w2.grad.norm().item())
#      print("-----")
# ================================
# Example 1 — Vanishing Gradient (Sigmoid)
# ================================

# Backprop chain (simplified 3-layer case):
# dL/dW1 = dL/dz3 * W3 * f'(z2) * W2 * f'(z1) * x

# Suppose:
# W2 ≈ 0.5
# W3 ≈ 0.5
# f'(z1) ≈ 0.2   (sigmoid derivative in non-central region)
# f'(z2) ≈ 0.2

# Gradient magnitude factor:
# 0.5 * 0.2 * 0.5 * 0.2 = 0.01

# So gradient shrinks to 1% in just 2 hidden layers.

# If repeated across many layers:
# (0.1)^10 = 1e-10
# → Early layer gradients ≈ 0
# → Training stalls

# This is vanishing gradient.
# ================================
# Example 2 — Exploding Gradient
# ================================

# Suppose:
# W2 ≈ 2
# W3 ≈ 2
# f'(z1) ≈ 1   (ReLU active)
# f'(z2) ≈ 1

# Gradient magnitude factor:
# 2 * 1 * 2 * 1 = 4

# With 10 layers:
# 2^10 = 1024

# Gradient becomes extremely large.
# → Loss oscillates
# → Updates unstable
# → Possible NaNs

# This is exploding gradient.
# ================================
# Example 3 — ReLU Gradient Flow
# ================================

# ReLU:
# f(x) = max(0, x)

# Derivative:
# f'(x) = 1  if x > 0
#         0  if x <= 0

# Case 1 — Neuron is ACTIVE (z > 0)

# Suppose:
# W2 ≈ 0.8
# W3 ≈ 0.9
# f'(z1) = 1
# f'(z2) = 1

# Gradient magnitude factor:
# 0.9 * 1 * 0.8 * 1 = 0.72

# No severe shrinking.
# Gradient flows relatively intact.
# This helps deep networks train.


# Case 2 — Neuron is INACTIVE (z <= 0)

# f'(z) = 0

# Backprop chain becomes:
# W3 * 0 * W2 * 0

# Entire gradient becomes 0.

# This neuron receives no update.
# If this keeps happening → "dying ReLU".
"""with xavier initialisation"""
import torch
torch.manual_seed(0)
x=torch.rand(200,2)
y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1) 
input_dim = 2
hidden_dim = 5
output_dim = 1
limit = (6 / (input_dim + hidden_dim)) ** 0.5
w1 = torch.empty(input_dim, hidden_dim).uniform_(-limit, limit)
w1.requires_grad_()
limit = (6 / (hidden_dim + output_dim)) ** 0.5
w2 = torch.empty(hidden_dim, output_dim).uniform_(-limit, limit)
w2.requires_grad_()
b1=torch.zeros(hidden_dim,requires_grad=True)
b2 = torch.zeros(output_dim, requires_grad=True)
lr = 0.1
for steps in range(1000):
    z1=x@w1+b1
    a1=torch.sigmoid(z1)
    z2=a1@w2+b2
    probs=torch.sigmoid(z2)
    loss=-(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
    loss.backward()
    with torch.no_grad():
        for param in [w1, b1, w2, b2]:
            param -= lr * param.grad
            param.grad.zero_()
    if steps % 100 == 0:
     print(f"step {steps}")
     print("loss:", loss.item())
     print("W1 grad norm:", w1.grad.norm().item())
     print("W2 grad norm:", w2.grad.norm().item())
     print("-----")


