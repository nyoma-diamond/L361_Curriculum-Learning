from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


# from grace:
import matplotlib.pyplot as plt
import numpy as np
from transfer_teacher_learning.femnist_tt.utils import curriculum_learning_loss, calculate_threshold, save_data, show_failed_imgs
from transfer_teacher_learning.femnist_tt.femnist import FemnistDataset
import copy

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################


# Use GPU on system if possible
warnings.filterwarnings("ignore", category=UserWarning)
if torch.cuda.is_available():
    print ("GPU CUDA")
    DEVICE = torch.device("cpu")
elif torch.backends.mps.is_available():
    print ("MPS device")
    DEVICE = torch.device("cpu")
else:
    print ("MPS device not found, using CPU")
    DEVICE = torch.device("cpu")


class FemnistNet(nn.Module):
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

def train(net, trainloader, config, cid):
    """Train the model on the training set."""

    net_teacher = copy.deepcopy(net)
    net.train() 

    criterion = torch.nn.CrossEntropyLoss(reduction='none') # @ N'yoma make sure to set reduction to none
    criterion_mean = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    losses = [] # @ N'yoma add things you want to save in here
    images_failed = []
    images_passed = []

    loss_threshold = calculate_threshold(net, 
                            criterion, 
                            trainloader, 
                            config["loss_threshold"], 
                            DEVICE,
                            config["threshold_type"], # change 0 for just flat num, 1, for percentile
                            config["percentile_type"]) # change "linear" for true percentile, "normal_unbiased" for normal


    for epoch in range(config['local_epochs']):
        batch_count = 0       # @ N'yoma add the batch count for saving files
        print("Epoch: "+str(epoch))

        
        for images, labels in trainloader:
            optimizer.zero_grad()

            images = images.to(DEVICE)      # @ N'yoma make sure to set images/labels to the device you're using
            labels = labels.to(DEVICE)

            trash_indices, keep_indices, loss_threshold, loss_indv = curriculum_learning_loss(net_teacher, 
                                                             criterion, 
                                                             images, 
                                                             labels, 
                                                             loss_threshold, 
                                                             DEVICE) 
            
            # @ N'yoma - saving data here, add more things you want to save if u need it
            for loss,lab in zip(loss_indv,labels):
              losses.append([loss.item(),loss_threshold,epoch,batch_count,lab.item()])

            for image,label, loss_ind in zip(images[trash_indices], labels[trash_indices], loss_indv[trash_indices]):
              images_failed.append([image,label,loss_ind, loss_threshold, epoch])
            for image,label, loss_ind in zip(images[keep_indices], labels[keep_indices], loss_indv[keep_indices]):
              images_passed.append([image,label,loss_ind, loss_threshold, epoch])
                        
            batch_count += 1
            criterion_mean(net(images[keep_indices.flatten(),:,:,:].to(DEVICE)), labels[keep_indices.flatten()].to(DEVICE)).backward()
            optimizer.step()
    
    save_data(losses, images_passed, images_failed, config['test_name'], cid, DEVICE)


def test(net, testloader):
  """Validate the model on the test set."""
  net.eval()

  criterion = torch.nn.CrossEntropyLoss()
  correct, total, loss = 0, 0, 0.0
  with torch.no_grad():
    for images, labels in testloader:
      outputs = net(images.to(DEVICE))
      loss += criterion(outputs, labels.to(DEVICE)).item()
      total += labels.size(0)
      # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
      correct += (torch.max(outputs.data, 1)[1] == labels.to(DEVICE)).sum().item() # add .toDevice to labels if using GPU
  return loss / len(testloader.dataset), correct / total

def load_data(cid):
  """Load CIFAR-10 (training and test set)."""
  trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
  # trainset = CIFAR10("./data", train=True, download=True, transform=trf)
  # testset = CIFAR10("./data", train=False, download=True, transform=trf)
  return train_loader, val_loader #DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

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
    train(self.net, self.trainloader, config, self.cid)
    return self.get_parameters(config={}), len(self.trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(self.net, self.testloader)
    return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: int) -> FlowerClient:
    # Load model and data (simple CNN, CIFAR-10)
    print("MADE CLIENT")
    net = FemnistNet().to(DEVICE)
    trainloader, testloader = load_data(cid)
    # train_loader = train_loaders[int(cid)]
    # val_loader = val_loaders[int(cid)]
    return FlowerClient(cid, net, trainloader, testloader).to_client()