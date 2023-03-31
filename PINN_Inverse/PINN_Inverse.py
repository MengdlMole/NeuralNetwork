# UTF-8 #
# python --version: Python 3.9.13 #
# cuda --version: 11.6.134 driver #
import torch
import numpy
import math


# define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.rand(
            1, requires_grad=True))  # the parameter gamma
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

    def show_gamma(self):
        return self.gamma

    # loss function depending on the ODEs
    def loss_of_eq(self, x, y_pred):
        pi = math.pi
        # right_item = torch.from_numpy(right_item0.astype(numpy.float32)).to(device)
        y_pred_grad = torch.autograd.grad(
            y_pred.sum(), x, create_graph=True)[0]
        # TODO: There depends what ODEs you want to solve
        loss = y_pred_grad - self.gamma * (1 + pi * torch.cos(pi * x / 2) / 2)
        return loss.t() @ loss

    def loss_of_initial(self, initial_point):
        initial_exact = exactSolution(initial_point)
        loss = self.linear_tanh_stack(
            initial_point.unsqueeze(1)) - initial_exact.unsqueeze(1)
        return loss.t() @ loss

    def loss_of_solution(self, x, y_pred):
        y_exact = exactSolution(x)
        loss = y_pred - y_exact
        return loss.t() @ loss

    def forward(self, x, initial_point):
        y_pred = self.linear_tanh_stack(x)
        return self.loss_of_eq(x, y_pred) + self.loss_of_initial(initial_point) + self.loss_of_solution(x, y_pred)
        # return self.loss_of_eq(x, y_pred) + self.loss_of_initial(model, initial_point)


# define ODEs
def exactSolution(x):
    pi = math.pi
    return x + torch.sin(pi * x / 2)


# train code
def train(device, model, optimizer, loss_fn, x, initial_point):
    for iter in range(30000):
        # Compute prediction error
        result = model(x.unsqueeze(1), initial_point)

        loss = loss_fn(result, torch.tensor(0, dtype=torch.float32).to(device))

        # loss = loss_fn(model, x, initial_point, y_pred, gamma)
        # loss.requires_grad_()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(f"iter {iter}, Loss: {loss.item()}")
            print(model.show_gamma())


# test code
def test(model):
    # TODO
    pass


def main():
    # Get cpu or gpu device for training
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    # device = "cuda"
    print(f"Using {device} device")

    x0 = numpy.linspace(0, 10, 1000)
    x = torch.from_numpy(x0.astype(numpy.float32)).to(device)
    x.requires_grad_()
    initial_point = torch.tensor(
        0, dtype=torch.float32).unsqueeze(0).to(device)

    # model.train()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # gamma = model.get_gamma().to(device)
    loss_fn = torch.nn.MSELoss()

    train(device, model, optimizer, loss_fn, x, initial_point)

    print(model.show_gamma())


if __name__ == '__main__':
    main()
    '''* TODO
    * layer size
    * 
    '''
