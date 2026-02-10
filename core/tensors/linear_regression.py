import torch

torch.manual_seed(0)

x = torch.randn(100,1)
true_W = 3.0
true_b = 2.0

y = true_W*x + true_b + 0.1*torch.randn(100,1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.1

for step in range(1000):

    y_pred = x*w + b
    loss = ((y_pred - y)**2).mean()

    loss.backward()

    if step % 100 == 0:
        print(step, loss.item(), w.grad.item())

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    # === Autograd + Training Loop Notes ===
# 1. Gradients accumulate by default in PyTorch (param.grad += new_grad),
#    so we must zero them after each optimizer step to avoid mixing gradients
#    from different parameter states. as Given the model as it is right now, how should I change the parameters.
#the weights were different
#the loss landscape was different
#possibly the data batch was different
# so we calc new grad and remove by zero grad as it is the slope which tells us how should we move to minimise the new loss for the new x rn
# 2. Leaf tensors = user-created parameters with requires_grad=True; only
#    leaf tensors receive .grad. Avoid reassigning params (e.g., w = w - lr*grad)
#    because that creates non-leaf tensors and breaks gradient tracking.
#    Use in-place updates under torch.no_grad(): w -= lr * w.grad
# 3. Optimizer updates are not part of the forward graph; wrap updates in
#    torch.no_grad() to prevent autograd from tracking parameter mutations.
# 4. detach() breaks graph history (use when creating new leaf params);
#    clone() copies data but keeps grad connection unless detached.
#    retain_graph=True is only for multiple backward passes on SAME graph.
# 5. Be careful with .data (bypasses autograd safety); prefer no_grad()+in-place ops.


    w.grad.zero_()
    b.grad.zero_()
## Which was easier to write?
"""the loss backward and updating weights """
## Which one helped me understand gradients?
"""a comment i wrote in line 36 """
## What does optimizer.step() replace?
""" updating the parameters"""

## What would break if I removed zero_grad()?
"""it would take the x as new operation and calc grad for that """
