# UTF-8 #
# python --version: Python 3.9.13 #
# cuda --version: 
# cuda --version: 11.6.134 driver #
import torch
import numpy
import math


# define model
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


# define ODEs
def exactSolution(x):
    pi = math.pi
    return x + torch.sin(pi*x/2)


# loss function depending on the ODEs
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()


    def loss_of_eq(self, x, y_pred):
        pi = math.pi
        # right_item = torch.from_numpy(right_item0.astype(numpy.float32)).to(device)
        y_pred_grad = torch.autograd.grad(y_pred.sum(), x, create_graph=True)[0]
        loss = y_pred_grad - (1 + pi*torch.cos(pi*x/2)/2)  # TODO: There depends what ODEs you want to solve
        return loss @ loss


    def loss_of_initial(self, model, initial_point):
        initial_exact = exactSolution(initial_point)
        loss = model(initial_point.unsqueeze(1)) - initial_exact.unsqueeze(1)
        return loss @ loss


    def forward(self, model, x, initial_point, y_pred):
        return self.loss_of_eq(x, y_pred) + self.loss_of_initial(model, initial_point)


# train code
def train(model, optimizer, loss_fn, x, initial_point):
    for iter in range(10000):
        # Compute prediction error
        y_pred = model(x.unsqueeze(1))

        loss = loss_fn(model, x, initial_point, y_pred)
        # loss.requires_grad_()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(f"iter {iter}, Loss: {loss.item()}")


# test code
def test(model):
    # TODO
    pass
    

def main():
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    # device = "cuda"
    print(f"Using {device} device")

    x0 = numpy.linspace(0, 10, 1000)
    x = torch.from_numpy(x0.astype(numpy.float32)).to(device)
    x.requires_grad_()
    initial_point = torch.tensor(0, dtype=torch.float32).unsqueeze(0).to(device)

    # model.train()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = Loss()

    train(model, optimizer, loss_fn, x, initial_point)

    
if __name__ == '__main__':
    main()
    '''* TODO
    * layer size
    * 
    '''
