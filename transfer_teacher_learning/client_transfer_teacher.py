from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import copy

# from grace:
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean,stdev

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################


# Use GPU on system if possible
warnings.filterwarnings("ignore", category=UserWarning)
if torch.cuda.is_available():
    print ("GPU CUDA")
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    print ("MPS device")
    DEVICE = torch.device("mps")
else:
    print ("MPS device not found, using CPU")
    DEVICE = torch.device("cpu")

def show_failed_imgs(new_images,new_labels,losses_failed):
    count = 0
    print(len(new_images))
    for images in new_images:
        if count > 3:
            break
        count +=1 
        # take first image
        image = images[0].to(DEVICE)
        # Reshape the image
        image = image.reshape(3,32,32).to(DEVICE)
        # Transpose the image
        image = image.permute(1, 2, 0).to(DEVICE)
        # Display the image
        plt.imshow(image.cpu())
        plt.show()

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

def train(net, trainloader, config, epochs):
    """Train the model on the training set."""

    net_teacher = copy.deepcopy(net)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_mean = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for k in range(epochs):
        # count = 0
        print("Epoch: "+str(k))
        for images, labels in trainloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            
            loss_indv = criterion(net_teacher(images), labels) # get individual losses
            # print(loss_indv)
            b = loss_indv >= config["loss_threshold"] # get indicies of which are larger than loss_threshold
            trash_indices = b.nonzero()
            d = loss_indv < config["loss_threshold"] # get indicies of which are larger than loss_threshold
            keep_indices = d.nonzero()
            # print(keep_indices)
            # print(count)
            # show_failed_imgs(images[trash_indices],labels[trash_indices],loss_indv[trash_indices])
            criterion_mean(net(images[keep_indices.flatten(),:,:,:]), labels[keep_indices.flatten()]).backward()
            # count += 1
            optimizer.step()


def test(net, testloader):
  """Validate the model on the test set."""
  criterion = torch.nn.CrossEntropyLoss()
  correct, total, loss = 0, 0, 0.0
  with torch.no_grad():
    for images, labels in testloader:
      outputs = net(images.to(DEVICE))
      loss += criterion(outputs, labels.to(DEVICE)).item()
      total += labels.size(0)
      correct += (torch.max(outputs.data, 1))[1] == labels.to(DEVICE).sum().item() # add .toDevice to labels if using GPU
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



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def __init__(self,
                 cid: int,
                 net: nn.Module,
                 trainloader: DataLoader,
                 testloader: DataLoader):
        
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader


  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(self.net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    self.net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(self.net, self.trainloader, config, config['local_epochs'])
    return self.get_parameters(config={}), len(self.trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(self.net, self.testloader)
    return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: int) -> FlowerClient:
    # Load model and data (simple CNN, CIFAR-10)
    print("MADE CLIENT")
    net = Net().to(DEVICE)
    trainloader, testloader = load_data()
    # train_loader = train_loaders[int(cid)]
    # val_loader = val_loaders[int(cid)]
    return FlowerClient(cid, net, trainloader, testloader).to_client()