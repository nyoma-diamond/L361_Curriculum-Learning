from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(local_net, global_net, trainloader, epochs, _lambda):
    """Train the model on the training set."""
    local_net.train()
    global_net.train()

    criterion = torch.nn.CrossEntropyLoss()
    local_optimizer = torch.optim.SGD(local_net.parameters(), lr=0.001, momentum=0.9)
    global_optimizer = torch.optim.SGD(global_net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            # Train the local model w/ our data, biased by the difference from the global model
            local_optimizer.zero_grad()
            criterion(local_net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            for param1, param2 in zip(local_net.parameters(), global_net.parameters()):
                param1.grad += _lambda * (param1 - param2)
            local_optimizer.step()

            # Train the global model with our data
            global_optimizer.zero_grad()
            criterion(global_net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            global_optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    local_net.eval()
    global_net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
local_net = Net().to(DEVICE)
global_net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in global_net.state_dict().items()]

    def set_parameters(self, net, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(global_net, parameters)
        train(local_net, global_net, trainloader, _lambda=1, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(global_net, parameters)
        local_loss, local_accuracy = test(local_net, testloader)
        global_loss, global_accuracy = test(global_net, testloader)
        return \
            float(global_loss), \
            len(testloader.dataset), \
            {
                'global_loss': float(global_loss),
                'global_accuracy': float(global_accuracy),
                'local_accuracy': float(local_accuracy)
            }

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())