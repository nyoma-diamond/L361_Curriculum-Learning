import os

import torch
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

CLIENT_MODEL_DIR = 'client_models'

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


def curriculum_learning_loss(net, loss_func, images, labels, loss_threshold, loss_type, percentile_type):
    '''

    Inputs: 
    - net: neural network
    - loss_func: torch.nn.CrossEntropyLoss(reduction='none')
        - Make sure to set reduction to none!
    - Images: batch of images
    - labels: batch of labels
    - loss_threshold: depending on what you enter as loss type, this can be actual loss value
        or the percentile value you want to test for your scenario
    - DEVICE: CPU/GPU
    - loss_type: This has the following options:
        - '0': use loss_threshold value as a singular number if a loss value is too high or low
        - '1': use loss_threshold value as a percentile in a normal distribution curve to see 
               if a loss value is too high or low 
    - percentile_type: check out this link for more info: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
        - 'Linear': true percentile
        - 'normal_unbiased': sampling from a normal distribution curve
    '''
    loss_indv = loss_func(net(images), labels)  # get individual losses

    if loss_type == 1:
        loss_threshold = np.percentile(loss_indv.data.cpu().numpy(), loss_threshold, method=percentile_type)

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
