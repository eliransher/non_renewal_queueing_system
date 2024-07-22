from fastai.vision.all import *
from fastbook import *
from sklearn.model_selection import train_test_split
matplotlib.rc('image', cmap='Greys')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import torch
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import torch
import torch.nn as nn

def queue_loss(predictions, targes):
    predictions = m(predictions)

    return ((torch.pow(torch.abs(predictions - targes[:, :]), 1)).sum(axis=1) +
            torch.max(torch.pow(torch.abs(predictions - targes[:, :]), 1), 1)[0]).sum()


def valid(dset_val, model):
    loss = 0
    for batch in dset_val:
        xb, yb = batch
        loss += queue_loss(model(xb), yb)
    return loss / len(dset_val)


def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    counter = 0
    curr_len = len(loss_list)
    if curr_len < n_last_steps:
        n_last_steps = curr_len

    inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
    for ind in inds_arr:
        if loss_list[-ind] > loss_list[-ind - 1]:
            counter += 1

    print(counter, n_last_steps)
    if counter / n_last_steps > failure_rate:
        return True

    else:
        return False


def compute_sum_error(valid_dl, model, return_vector, max_err=0.05, display_bad_images=False):
    with torch.no_grad():
        bad_cases = {}
        for ind, batch in enumerate(valid_dl):

            xb, yb = batch
            predictions = m(model(xb[:, :]))

            curr_errors = torch.sum(torch.abs((predictions - yb[:, :])), axis=1)
            bad_dists_inds = (curr_errors > max_err).nonzero(as_tuple=True)[0]
            # if display_bad_images:
            #     print_bad_examples(bad_dists_inds, predictions, yb)
            if ind == 0:
                sum_err_tens = torch.sum(torch.abs((predictions - yb[:, :])), axis=1)
            else:
                sum_err_tens = torch.cat((sum_err_tens, curr_errors), axis=0)
    if return_vector:
        return torch.mean(sum_err_tens), sum_err_tens
    else:
        return torch.mean(sum_err_tens)



def main():
    path = '/scratch/eliransc/pkl_training'
    files = os.listdir(path)

    path = '/scratch/eliransc/pkl_training'
    files = os.listdir(path)
    m_data, y_data = pkl.load(open(os.path.join(path, files[0]), 'rb'))

    path = '/scratch/eliransc/pkl_training'
    files = os.listdir(path)
    m_data, y_data = pkl.load(open(os.path.join(path, files[0]), 'rb'))

    m_data_valid = m_data[: 1000, :]
    y_data_valid = y_data[: 1000, :]

    m_data = m_data[1000:, :]
    y_data = y_data[1000:, :]

    num_moms = 5
    num_corrs = 3

    m_data = torch.tensor(m_data)
    y_data = torch.tensor(y_data)

    m_data_valid = torch.tensor(m_data_valid)
    y_data_valid = torch.tensor(y_data_valid)

    m_data = m_data.float()
    y_data = y_data.float()

    m_data_valid = m_data_valid.float()
    y_data_valid = y_data_valid.float()

    # Construct dataset
    dset = list(
        zip(torch.cat((m_data[:, :num_moms], m_data[:, 10:10 + num_corrs], m_data[:, 15:15 + num_moms]), 1), y_data))
    valid_dset = list(zip(torch.cat(
        (m_data_valid[:, :num_moms], m_data_valid[:, 10:10 + num_corrs], m_data_valid[:, 15:15 + num_moms]), 1),
                          y_data_valid))
    dl = DataLoader(dset, batch_size=128)
    valid_dl = DataLoader(valid_dset, batch_size=128)
    m = nn.Softmax(dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Net(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(2 * num_moms + num_corrs, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 100)
            self.fc4 = nn.Linear(100, 200)
            self.fc5 = nn.Linear(200, 200)
            self.fc6 = nn.Linear(200, 350)
            self.fc7 = nn.Linear(350, 500)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = self.fc7(x)
            return x  # F.log_softmax(x,dim=1)

    net = Net().to(device)
    weight_decay = 5
    curr_lr = 0.01
    EPOCHS = 240
    dl.to(device)
    valid_dl.to(device)
    now = datetime.now()
    lr_second = 0.99
    lr_first = 0.75
    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    print('curr time: ', current_time)

    # Construct dataset

    optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                           weight_decay=(1 / 10 ** weight_decay))  # paramters is everything adjustable in model

    loss_list = []
    valid_list = []
    compute_sum_error_list = []

    for epoch in tqdm(range(EPOCHS)):
        t_0 = time.time()
        for data in dl:
            X, y = data

            if torch.sum(torch.isinf(X)).item() == 0:
                net.zero_grad()
                output = net(X)
                loss = queue_loss(output, y)  # 1 of two major ways to calculate loss
                loss.backward()
                optimizer.step()
                net.zero_grad()

        loss_list.append(loss.item())
        valid_list.append(valid(valid_dl, net).item())
        compute_sum_error_list.append(compute_sum_error(valid_dl, net, False).item())

        if len(loss_list) > 3:
            if check_loss_increasing(valid_list):
                curr_lr = curr_lr * lr_first
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                print(curr_lr)
            else:
                curr_lr = curr_lr * lr_second
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                print(curr_lr)

        print("Epoch: {}, Training: {:.5f}, Validation : {:.5f}, Valid_sum_err: {:.5f},Time: {:.3f}".format(epoch,
                                                                                                            loss.item(),
                                                                                                            valid_list[
                                                                                                                -1],
                                                                                                            compute_sum_error_list[
                                                                                                                -1],
                                                                                                            time.time() - t_0))
        # torch.save(net.state_dict(), '../gg1_models/pytorch_g_g_1_true_moms_new_data_' + setting_string +'_'+ str(
        #     current_time) + '.pkl')
        # pkl.dump((loss_list, valid_list, compute_sum_error_list),
        #          open('../gg1_models/losts_'   + '_new_data_'+ setting_string+'_'+ str(current_time) + '.pkl', 'wb'))
