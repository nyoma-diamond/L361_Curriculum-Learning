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
    SELF_PACED = 0
    TRANSFER_TEACHER = 1


def get_device(log=False):
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


def curriculum_learning_loss(
        net: nn.Module,
        loss_func: _Loss,
        images: torch.Tensor,
        labels: torch.Tensor,
        loss_threshold: float,
        loss_type: ThresholdType = ThresholdType.PERCENTILE,
        percentile_type: str = 'linear'
    ):
    '''

    Inputs: 
    - net: neural network
    - loss_func: loss function for computing threshold
        - Make sure to that no reduction is performed!
    - Images: batch of images
    - labels: batch of labels
    - loss_threshold: depending on what you enter as loss type, this can be actual loss value
        or the percentile value you want to test for your scenario
    - DEVICE: CPU/GPU
    - loss_type: This has the following options:
        - ThresholdType.PERCENTILE: use loss_threshold value as a singular number if a loss value is too high or low
        - ThresholdType.DIRECT_VALUE: use loss_threshold value as a percentile in a normal distribution curve to see
                                      if a loss value is too high or low
    - percentile_type: check out this link for more info: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
        - 'linear': true percentile
        - 'normal_unbiased': sampling from a normal distribution curve
    '''
    assert loss_func.reduction == 'none', 'loss function must have the reduction type "none".'

    loss_indv = loss_func(net(images), labels)  # get individual losses

    match loss_type:
        case ThresholdType.PERCENTILE:
            loss_threshold = np.percentile(loss_indv.data.cpu().numpy(), loss_threshold, method=percentile_type)
        case ThresholdType.QUANTILE:
            loss_threshold = np.quantile(loss_indv.data.cpu().numpy(), loss_threshold, method=percentile_type)


    # print(loss_indv)
    b = loss_indv >= loss_threshold  # get indicies of which are larger than loss_threshold
    trash_indices = b.nonzero()
    d = loss_indv < loss_threshold  # get indicies of which are larger than loss_threshold
    keep_indices = d.nonzero()

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
