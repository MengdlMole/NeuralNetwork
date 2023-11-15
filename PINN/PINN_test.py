# ODE:
# u + \Delta u = f = (x^2+1)/2
# u(-1) = u(1) = 0
# u = (x^2-1)/2

# 导入库：
import torch
import torch.nn.functional as F
import numpy as np
import math

# 定义神经网络：
class PINNs(torch.nn.Module):
    def __init__(self, m):
        super(PINNs, self).__init__()
        self.linear1 = torch.nn.Linear(1, m)
        self.linear2 = torch.nn.Linear(m, m)
        self.linear3 = torch.nn.Linear(m, 1)
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1, m),
            torch.nn.ReLU(),
            torch.nn.Linear(m, m),
            torch.nn.ReLU(),
            torch.nn.Linear(m, 1)
        )
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(1, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, m),
            torch.nn.Tanh(),
            torch.nn.Linear(m, 1)
        )

    def forward(self, x):
        output1 = F.tanh(self.linear1(x))
        output2 = F.tanh(self.linear2(output1))
        output3 = self.linear3(output2)
        return output3
        # solution = self.linear_tanh_stack(x)
        # return solution  

# 确定训练参数及初始化模型：
m = 100
learing_rate = 0.01
iterations = 10000
model = PINNs(m)
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

# 确定训练区域：
points = np.arange(-1, 1, 0.1)
# x = torch.from_numpy(points.astype(np.float32)).unsqueeze(1)
x = torch.tensor(points, dtype=torch.float32).unsqueeze(1)
dim = x.size()[0]

# 训练：
for k in range(iterations):
    loss = torch.zeros(1, dtype=torch.float32)
    lb = -np.ones(1)
    rb = np.ones(1)
    left_boundary = torch.tensor(lb, dtype = torch.float32).requires_grad_(True)
    right_boundary = torch.tensor(rb, dtype = torch.float32).requires_grad_(True)
    loss = loss + model(left_boundary) ** 2 + model(right_boundary) ** 2
    for i in range(dim):
        point = torch.tensor([x[i]]).requires_grad_(True)
        u = model(point)
        # u_x = torch.autograd.grad(u.sum(), point, create_gr/aph=True)[0]
        u_x = torch.autograd.grad(u, point, create_graph=True, retain_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, point, create_graph=True, retain_graph = True)[0]
        loss  = loss + (u+u_xx-(point**2+1)/2)**2/dim
    # if loss < 0.00001:
        # break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (k+1) % 1000 == 0:
        print(loss)
