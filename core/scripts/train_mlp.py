# import torch
# import torch.nn as nn
# torch.manual_seed(0)
# x=torch.rand(200,2)
# y=(x[:,0]+x[:,1]>0).float().unsqueeze(1)
# lr=0.1
# model=nn.Sequential(
#     nn.Linear(2,5),
#     nn.ReLU(),
#     nn.Linear(5,1)
# )
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# for steps in range(1000):
#     logits=model(x)
#     loss=criterion(logits,y)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if steps % 100 == 0:
#         print(f"step {steps} | loss {loss.item():.4f}")
"""
Absolutely. Here’s a concise, high-level summary of everything you’ve asked so far:

---

1. **Input `x` and label `y`:**

   * `x` is a tensor of shape `(samples, features)` (e.g., `(200,1)` or `(200,2)`).
   * `y` is the label computed from `x` (e.g., `y = (x>0)` or `y = (x[:,0]+x[:,1]>0)`), converted to float and reshaped `(samples,1)` for PyTorch.

2. **Logistic regression vs neural network weights:**

   * Logistic regression: 1 feature → 1 weight (`w.shape=(1,)`) and 1 bias → simple scalar operations.
   * Multi-layer network: each neuron in a layer needs **weights for all inputs** → `weight.shape = (input_dim, hidden_dim)`; bias = 1 per neuron.

3. **Two weight matrices (`w1`, `w2`):**

   * `w1`: input → hidden layer
   * `w2`: hidden → output layer
   * Each layer learns **different transformations**, so separate weights are required.

4. **Matrix multiplication (`@`):**

   * Computes linear combinations for all neurons efficiently: `(batch_size, in_features) @ (in_features, out_features) → (batch_size, out_features)`
   * Needed because each neuron sums over all its input features.

5. **Activation (`a1 = ReLU(z1)`):**

   * Introduces nonlinearity so the network can learn complex patterns.
   * Output of hidden neurons (`a1`) is then fed to the next layer with `w2`.

6. **Hidden neurons (`hidden_dim`):**

   * You can define how many hidden neurons to use.
   * More neurons → more expressive network; fewer → simpler network.
   * Biases exist per neuron so each can shift activation independently.

7. **BCEWithLogitsLoss vs manual sigmoid + BCE:**

   * Combines sigmoid and binary cross-entropy in one numerically stable operation.
   * Avoids gradient instability that occurs if logits are very large or very small.

8. **Zero initialization vs random initialization:**

   * Logistic regression: zero init is fine (only one weight vector, no symmetry issue).
   * Deep networks: zero init is bad → all neurons in a layer stay identical (symmetry problem), and learning is blocked.
   * Random init breaks symmetry → neurons learn different features.

9. **Shape rules summary:**

| Layer           | Weight shape               | Bias shape      | Input/output shape                            |
| --------------- | -------------------------- | --------------- | --------------------------------------------- |
| Input → Hidden  | `(input_dim, hidden_dim)`  | `(hidden_dim,)` | `(batch, input_dim)` → `(batch, hidden_dim)`  |
| Hidden → Output | `(hidden_dim, output_dim)` | `(output_dim,)` | `(batch, hidden_dim)` → `(batch, output_dim)` |


"""


"""using sigmoid"""
# import torch
# import torch.nn as nn
# torch.manual_seed(0)
# x=torch.rand(200,2)
# y=(x[:,0]+x[:,1]>0).float().unsqueeze(1)
# lr=0.1
# model=nn.Sequential(
#     nn.Linear(2,5),
#     nn.Sigmoid(),
#     nn.Linear(5,1)
# )
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# for steps in range(1000):
#     logits=model(x)
#     loss=criterion(logits,y)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if steps % 100 == 0:
#         print(f"step {steps} | loss {loss.item():.4f}")
"""using xavier initialisation """
import torch
import torch.nn as nn

torch.manual_seed(0)

x = torch.rand(200, 2)
y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)

lr = 0.1

model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Sigmoid(),
    nn.Linear(5, 1)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for steps in range(1000):
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if steps % 100 == 0:
        print(f"step {steps} | loss {loss.item():.4f}")
