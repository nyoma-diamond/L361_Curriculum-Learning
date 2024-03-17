# Modified from https://flower.ai/
import sys
from collections import OrderedDict
from functools import partial
from typing import Tuple, Dict, Optional, Callable
import warnings

import flwr as fl
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from femnist import FemnistDataset, FemnistNet
from utils import *

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings('ignore', category=UserWarning)

# Use GPU on system if possible
DEVICE = get_device()

def train(local_net: nn.Module, global_net: nn.Module, train_loader: DataLoader, config: dict, cid: int):
    """Train the model on the training set."""
    if type(local_net) is not type(global_net):
        raise TypeError(f'Ditto training expects local_net and global_net to be the same type. Got {type(local_net)} and {type(global_net)}.')

    local_net.train()
    global_net.train()

    curriculum_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_mean = torch.nn.CrossEntropyLoss(reduction='mean')

    # TODO: config for optimizer parameters
    local_optimizer = torch.optim.SGD(local_net.parameters(), lr=0.01, momentum=0.9)
    global_optimizer = torch.optim.SGD(global_net.parameters(), lr=0.01, momentum=0.9)

    losses = []

    for epoch in range(config['local_epochs']):
        for batch_i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Train the local model w/ our data, biased by the difference from the global model
            local_optimizer.zero_grad()

            trash_indices, keep_indices, loss_threshold, loss_indv = curriculum_learning_loss(
                global_net,  # TODO: global net or local net?
                curriculum_criterion,
                images,
                labels,
                config['loss_threshold'],
                config['threshold_type'],  # change 0 for just flat num, 1, for percentile
                config['percentile_type'])  # change 'linear' for true percentile, 'normal_unbiased' for normal

            for loss in loss_indv:
              losses.append([loss.item(), loss_threshold, epoch, batch_i])

            # Update local model
            local_loss = criterion_mean(local_net(images[keep_indices.flatten(),:,:,:]), labels[keep_indices.flatten()])
            local_loss.backward()
            for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
                local_param.grad += config['lambda'] * (local_param - global_param)
            local_optimizer.step()

            # Update global model
            global_optimizer.zero_grad()
            global_loss = criterion_mean(global_net(images[keep_indices.flatten(),:,:,:]), labels[keep_indices.flatten()])
            global_loss.backward()
            global_optimizer.step()

        save_data(losses, config['test_name'], cid)

def test(net: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Validate the model on the test set."""
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
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
                 _lambda: float):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.local_net = FemnistNet().to(DEVICE)
        self.global_net = FemnistNet().to(DEVICE)

        self._lambda = _lambda

    def save_local_model(self):
        torch.save(self.local_net.state_dict(), f'{CLIENT_MODEL_DIR}/{self.cid}.pth')

    def load_local_model(self, parameters: Optional[NDArrays]):
        try:
            self.local_net.load_state_dict(torch.load(f'{CLIENT_MODEL_DIR}/{self.cid}.pth'))
        except Exception as e:  # this will always occur on the first round
            print(f'Could not load local model for client {self.cid} due to {type(e).__name__}. Copying provided parameters to local model. This IS expected for a client\'s first round', file=sys.stderr)
            if parameters is not None:
                self.set_parameters(self.local_net, parameters)
            else:
                print('WARNING: No parameters provided to initialize local model. Using new randomized local model. This behavior is unexpected.', file=sys.stderr)
                self.local_net = FemnistNet().to(DEVICE)

    def get_parameters(self, config: Config) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]

    def set_parameters(self, net: nn.Module, parameters: NDArrays):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if 'lambda' in config and config['lambda'] is not None:
            self._lambda = config['lambda']
        else:
            config['lambda'] = self._lambda

        self.set_parameters(self.global_net, parameters)
        self.load_local_model(parameters)

        train(self.local_net, self.global_net, self.train_loader, config, self.cid)

        self.save_local_model()
        # TODO: report training metrics
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if 'lambda' in config and config['lambda'] is not None:
            self._lambda = config['lambda']
        else:
            config['lambda'] = self._lambda

        self.set_parameters(self.global_net, parameters)
        self.load_local_model(parameters)

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


def ditto_client_fn(cid: int, _lambda: float = 1.0) -> DittoClient:
    """Ditto client generator"""
    train_loader = DataLoader(
        FemnistDataset(client=cid, split='train', transform=ToTensor()),
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        FemnistDataset(client=cid, split='test', transform=ToTensor()),
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    return DittoClient(cid, train_loader, val_loader, _lambda=_lambda)

def ditto_client_fn_generator(_lambda: float = 1.0) -> Callable[[int], DittoClient]:
    return partial(ditto_client_fn, _lambda=_lambda)
