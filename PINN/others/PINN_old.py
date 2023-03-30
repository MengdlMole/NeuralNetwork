import torch
import numpy
import math


# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 1)
        )
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(1, 40),
            torch.nn.Tanh(),
            torch.nn.Linear(40, 40),
            torch.nn.Tanh(),
            torch.nn.Linear(40, 1)
        )


    def forward(self, x):
        solution = self.linear_tanh_stack(x)
        return solution


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()


    def forward(self, loss_pred, loss_target):
        return loss_pred - loss_target


# Define ODE problem
def exactSolution(x):
    pi = math.pi
    return x + torch.sin(pi*x/2)


def rightODE(x):
    pi = math.pi
    return 1 + pi*torch.cos(pi*x/2)/2


# Train the model
def train(x, right_item, initial_point, initial_exact, device):
    # model.train()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = torch.nn.MSELoss()
    for iter in range(10000):
        # Compute prediction error
        y_pred = model(x.unsqueeze(1))
        y_pred_grad = torch.autograd.grad(y_pred.sum(), x, grad_outputs=None, retain_graph=None, create_graph=True, allow_unused=False, is_grads_batched=False)[0]

        loss_of_initial = model(initial_point.unsqueeze(1)) - initial_exact.unsqueeze(1)
        loss_of_eq = y_pred_grad - right_item
        
        loss = loss_of_initial @ loss_of_initial + loss_of_eq @ loss_of_eq
        # loss = loss_of_initial @ loss_of_initial.t() + loss_of_eq @ loss_of_eq.t()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(f"iter {iter}, Loss: {loss.item()}")
    


def main():
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    x0 = numpy.linspace(0, 10, 1000)
    x = torch.from_numpy(x0.astype(numpy.float32)).to(device)
    x.requires_grad_()
    # x = 10*numpy.random.rand(1000)
    # right_item = rightODE(x)

    # exact = exactSolution(x)
    # x = x.to(device)
    # right_item = right_item.to(device)
    # right_item.requires_grad_(True)
    # exact = exact.to(device)
    initial_point = torch.tensor(0, dtype=torch.float32).unsqueeze(0).to(device)
    initial_exact = exactSolution(initial_point)
    # initial_point = initial_point.to(device)
    # initial_exact = initial_exact.to(device)


    # model.train()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = Loss().to(device)

    for iter in range(10000):
        right_item = rightODE(x)
        # Compute prediction error
        y_pred = model(x.unsqueeze(1))
        y_pred_grad = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0]

        loss_of_initial = model(initial_point.unsqueeze(1)) - initial_exact.unsqueeze(1)
        loss_of_eq = y_pred_grad - right_item
        
        # loss = loss_fn(y_pred, exact.unsqueeze(1))
        # loss = loss_fn(loss_of_initial @ loss_of_initial + loss_of_eq @ loss_of_eq, torch.tensor(0, dtype=torch.float32).to(device))
        loss = loss_of_initial @ loss_of_initial + loss_of_eq @ loss_of_eq
        # loss = torch.nn.functional.mse_loss(loss_of_eq, torch.zeros_like(loss_of_eq))
        # loss = loss_fn(loss_of_initial @ loss_of_initial + loss_of_eq @ loss_of_eq, torch.tensor(0, dtype=torch.float32).to(device))
        # loss = loss_of_initial @ loss_of_initial.t() + loss_of_eq @ loss_of_eq.t()
        loss.requires_grad_()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(f"iter {iter}, Loss: {loss.item()}")


if __name__ == '__main__':
    main()
