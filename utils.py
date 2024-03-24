import os
from enum import Enum

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.modules.loss import _Loss

# import matplotlib.pyplot as plt

CLIENT_MODEL_DIR = 'client_models'

class ThresholdType(Enum):
    DIRECT_VALUE = 0
    PERCENTILE = 1
    QUANTILE = 2

class CurriculumType(Enum):
    NONE = 0
    SELF_PACED = 1
    TRANSFER_TEACHER = 2


def get_device(log=False):
    """
    Get the device for torch to use

    :param log: print out which device was selected
    """
    if torch.cuda.is_available():
        if log:
            print('Using CUDA')
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        if log:
            print('Using MPS')
        return torch.device('mps')
    else:
        if log:
            print('Using CPU')
        return torch.device('cpu')

DEVICE = get_device()


def compute_curriculum(
        teacher_net: nn.Module,
        loss_func: _Loss,
        train_loader: torch.utils.data.DataLoader,
        loss_threshold: float,
        threshold_type: ThresholdType,
        percentile_type: str = 'linear',
        device: torch.device = DEVICE
    ):
    """
    Compute curriculum for the current epoch

    :param net: model to compute threshold on
    :param loss_func: loss function to compute threshold with
    :param train_loader: dataloader containing training data to compute threshold on
    :param loss_threshold: desired base threshold to compute cutoff from
    :param threshold_type: ThresholdType indicating whether to use direct loss or percentile/quantile cutoff
    :param percentile_type: percentile computation method; passed to np.percentile/np.quantile.
                            expected: 'linear' for true cutoff, 'normal_unbiased' for normal approximated cutoff
    :param device: CPU/GPU device to compute on (should be same as net's device)

    :return: indices of tensors beyond threshold, indices of tensors within threshold, computed loss cutoff threshold, observed losses
    """
    assert loss_func.reduction == 'none', 'loss function must have the reduction type "none".'

    loss_indv = []

    if threshold_type in [ThresholdType.PERCENTILE, ThresholdType.QUANTILE]:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            losses = loss_func(teacher_net(images), labels).detach().cpu()
            loss_indv.append(losses)

        match threshold_type:
            case ThresholdType.PERCENTILE:
                loss_threshold = np.percentile(loss_indv, loss_threshold, method=percentile_type)
            case ThresholdType.QUANTILE:
                loss_threshold = np.quantile(loss_indv, loss_threshold, method=percentile_type)
            case _:
                raise Exception('Invalid ThresholdType. Something went VERY wrong')

    trash_indices = []
    keep_indices = []

    for batch in loss_indv:
        b = batch >= loss_threshold  # get indicies of which are larger than loss_threshold
        trash_indices.append(b.nonzero().flatten())
        d = batch < loss_threshold  # get indicies of which are larger than loss_threshold
        keep_indices.append(d.nonzero().flatten())

    return trash_indices, keep_indices, loss_threshold, loss_indv


def save_data(losses, test_name, cid):
    folder_name = f'results/{test_name}/cid_{cid}'
    round = 0
    x = pd.DataFrame(losses, columns=['sample_loss',
                                      'loss_threshold_of_batch',
                                      'epoch',
                                      'batch_count'])

    if not os.path.exists(f'{folder_name}/round_0'):
        os.makedirs(f'{folder_name}/round_0')
    else:
        s = os.listdir(folder_name)
        round = int(s[0][-1]) + 1
        if not os.path.exists(f'{folder_name}/round_{round}'):
            os.makedirs(f'{folder_name}/round_{round}')
    x.to_csv(f'{folder_name}/round_{round}/losses.csv')
    return
