# Modified from https://flower.ai/

from collections import OrderedDict
from typing import Tuple, Dict

import flwr as fl
from flwr.common.typing import Config, NDArrays, Scalar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils import *
from femnist import FEMNIST

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

# warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """Simple CNN model for FEMNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(local_net: nn.Module, global_net: nn.Module, train_loader: DataLoader, _lambda: float, epochs: int):
    """Train the model on the training set."""
    if type(local_net) is not type(global_net):
        raise TypeError(f'Ditto training expects local_net and global_net to be the same type. Got {type(local_net)} and {type(global_net)}.')

    local_net.train()
    global_net.train()

    criterion = torch.nn.CrossEntropyLoss()
    local_optimizer = torch.optim.SGD(local_net.parameters(), lr=0.01, momentum=0.9)
    global_optimizer = torch.optim.SGD(global_net.parameters(), lr=0.01, momentum=0.9)
    for i in range(epochs):
        for images, labels in train_loader:
            # Train the local model w/ our data, biased by the difference from the global model
            local_optimizer.zero_grad()
            criterion(local_net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
                local_param.grad += _lambda * (local_param - global_param)
            local_optimizer.step()

            # Train the global model with our data
            global_optimizer.zero_grad()
            criterion(global_net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            global_optimizer.step()

def test(net: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Validate the model on the test set."""
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(test_loader.dataset), correct / total


# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Define Ditto client
class DittoClient(fl.client.NumPyClient):
    # Modified from https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html
    def __init__(self,
                 cid: int,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 _lambda: float, epochs_per_round: int):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.local_net = Net().to(DEVICE)
        self.global_net = Net().to(DEVICE)

        # TODO: these may be altered via `config` during operation
        self._lambda = _lambda
        self.epochs_per_round = epochs_per_round

    def save_local_model(self):
        torch.save(self.local_net.state_dict(), f'{CLIENT_MODEL_DIR}/{self.cid}.pth')

    def load_local_model(self):
        try:
            self.local_net.load_state_dict(torch.load(f'{CLIENT_MODEL_DIR}/{self.cid}.pth'))
        except Exception as e:  # this will always occur on the first round
            print(f'Could not load local model for client {self.cid} due to {type(e).__name__}. Using new model.')

    def get_parameters(self, config: Config) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]

    def set_parameters(self, net: nn.Module, parameters: NDArrays):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(self.global_net, parameters)
        self.load_local_model()

        train(self.local_net, self.global_net, self.train_loader, self._lambda, self.epochs_per_round)

        self.save_local_model()
        # TODO: report training metrics
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(self.global_net, parameters)
        self.load_local_model()
        local_loss, local_accuracy = test(self.local_net, self.val_loader)
        global_loss, global_accuracy = test(self.global_net, self.val_loader)
        return \
            float(global_loss), \
            len(self.val_loader.dataset), \
            {
                'local_loss': float(local_loss),
                'global_accuracy': float(global_accuracy),
                'local_accuracy': float(local_accuracy)
            }



def ditto_client_fn(cid: int) -> DittoClient:
    """Ditto client generator"""
    train_loader = DataLoader(
        FEMNIST(client=cid, split='train', transform=ToTensor()),
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        FEMNIST(client=cid, split='test', transform=ToTensor()),
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    return DittoClient(cid, train_loader, val_loader, 1, 100)
