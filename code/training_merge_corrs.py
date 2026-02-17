import os
import sys

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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device



def scv_partion(df):
    # --- binning ---
    def bin_scv(x):
        if pd.isna(x): return np.nan
        return "<3" if x < 3 else ">3"

    def bin_mom1ratio(x):
        if pd.isna(x): return np.nan
        return "<0.25" if x < 0.25 else ">0.25"

    def bin_rho(x):
        if pd.isna(x): return np.nan
        if x < -0.25:
            return "<-0.25"
        elif x < 0:
            return "(-0.25,0)"
        elif x < 0.25:
            return "(0,0.25)"
        elif x < 1:
            return "(0.25,1)"
        else:
            return ">=1"  # outside requested bins

    df["SCV1_bin"] = df["SCV1"].apply(bin_scv)
    df["SCV2_bin"] = df["SCV2"].apply(bin_scv)
    df["rho1_bin"] = df["rho1"].apply(bin_rho)
    df["rho2_bin"] = df["rho2"].apply(bin_rho)
    df["mom1ratio_bin"] = df["mom1ratio"].apply(bin_mom1ratio)

    err_cols = ["err2", "err3", "err4", "err5", "err6", "err7", "err8"]

    # ----------------------------
    # 1) Split by combined (rho1, rho2, mom1ratio)
    # ----------------------------
    group_rhos = ["rho1_bin", "rho2_bin", "mom1ratio_bin"]

    summary_rhos = (
        df.groupby(group_rhos, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_rhos.columns = [f"{c}_{stat}" for c, stat in summary_rhos.columns]
    summary_rhos = summary_rhos.reset_index()

    # Include empty groups (cartesian product of requested bins)
    rho_bins = ["<-0.25", "(-0.25,0)", "(0,0.25)", "(0.25,1)"]
    mom_bins = ["<0.25", ">0.25"]

    full_idx_rhos = pd.MultiIndex.from_product([rho_bins, rho_bins, mom_bins], names=group_rhos)
    summary_rhos_full = (
        df[df["rho1_bin"].isin(rho_bins) & df["rho2_bin"].isin(rho_bins)]
        .groupby(group_rhos, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_rhos_full.columns = [f"{c}_{stat}" for c, stat in summary_rhos_full.columns]
    summary_rhos_full = summary_rhos_full.reindex(full_idx_rhos).reset_index()
    summary_rhos_full[[f"{c}_count" for c in err_cols]] = summary_rhos_full[[f"{c}_count" for c in err_cols]].fillna(
        0).astype(int)

    # ----------------------------
    # 2) Split by combined (SCV1, SCV2, mom1ratio)
    # ----------------------------
    group_scvs = ["SCV1_bin", "SCV2_bin", "mom1ratio_bin"]

    summary_scvs = (
        df.groupby(group_scvs, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_scvs.columns = [f"{c}_{stat}" for c, stat in summary_scvs.columns]
    summary_scvs = summary_scvs.reset_index()

    # Include empty groups
    scv_bins = ["<3", ">3"]
    full_idx_scvs = pd.MultiIndex.from_product([scv_bins, scv_bins, mom_bins], names=group_scvs)

    summary_scvs_full = (
        df.groupby(group_scvs, dropna=False)[err_cols]
        .agg(["count", "mean"])
    )
    summary_scvs_full.columns = [f"{c}_{stat}" for c, stat in summary_scvs_full.columns]
    summary_scvs_full = summary_scvs_full.reindex(full_idx_scvs).reset_index()
    summary_scvs_full[[f"{c}_count" for c in err_cols]] = summary_scvs_full[[f"{c}_count" for c in err_cols]].fillna(
        0).astype(int)

    # ---- outputs ----
    # summary_rhos_full: includes empty rho1/rho2/mom1ratio groups
    # summary_scvs_full: includes empty SCV1/SCV2/mom1ratio groups

    return summary_scvs_full, summary_rhos_full


def check_test_corrs(loader_valid, net_moms, init_ind, num_arrival_moms):
    all_errs = []
    for X1, y1 in loader_valid:
        X1 = X1.float()
        y1 = y1.float()
        X1 = X1.to(device)
        y1 = y1.to(device)
        # X1 = X1[:, 0, :]
        # y1 = y1[:, 0, :]
        SCV_1 = (torch.exp(X1[:, 1]) - torch.exp(X1[:, 0]) ** 2) / torch.exp(X1[:, 0]) ** 2
        SCV_2 = (torch.exp(X1[:, init_ind+1]) - torch.exp(X1[:, init_ind]) ** 2) / torch.exp(X1[:, init_ind]) ** 2
        mom1_ratio = torch.exp(X1[:, 0])
        rho_1 = X1[:, num_arrival_moms]
        rho_2 = X1[:, init_ind+ 2*num_arrival_moms]

        preds = net_moms(X1)
        # curr_errs = 100 * torch.abs((torch.exp(preds[:, :]) - torch.exp(y1[:, :])) / torch.exp(y1[:, :])).mean(axis=0)
        curr_torch =  torch.abs(preds[:, :] - y1[:, :])

        curr_torch = torch.concatenate((curr_torch, SCV_1.reshape(-1, 1)), axis=1)
        curr_torch = torch.concatenate((curr_torch, SCV_2.reshape(-1, 1)), axis=1)

        curr_torch = torch.concatenate((curr_torch, rho_1.reshape(-1, 1)), axis=1)
        curr_torch = torch.concatenate((curr_torch, rho_2.reshape(-1, 1)), axis=1)

        curr_torch = torch.concatenate((curr_torch, mom1_ratio.reshape(-1, 1)), axis=1)

        all_errs.append(curr_torch)

    tot_mom_res = torch.vstack(all_errs)

    with torch.no_grad():
        sumres = pd.DataFrame(tot_mom_res.cpu(),
                              columns=['err1', 'err2', 'err3', 'err4', 'err5', 'err6', 'err7', 'err8',
                                       'SCV1', 'SCV2', 'rho1', 'rho2', 'mom1ratio'])

    return sumres





def valid(loader_valid, model):
    loss = 0

    for X, y in loader_valid:
        X = X.float()
        y = y.float()
        X = X.to(device)
        y = y.to(device)
        # X = X[:, 0, :]
        # y = y[:, 0, :]
        loss += depart_loss_correlation(model(X), y)
    return loss / len(loader_valid)




def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    counter = 0
    curr_len = len(loss_list)
    if curr_len < n_last_steps:
        n_last_steps = curr_len

    inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
    for ind in inds_arr:
        if loss_list[-ind] > loss_list[-ind - 1]:
            counter += 1

    # print(counter, n_last_steps)
    if counter / n_last_steps > failure_rate:
        return True

    else:
        return False









def depart_loss(preds, target, num_moms_depart=5):
    weights = torch.flip(torch.arange(1, num_moms_depart), dims=(0,))
    weights = weights.to(device)
    # loss_arrive = torch.abs((preds[:,:num_moms_depart]-target[:,:num_moms_depart])/target[:,:num_moms_depart]) #  weights*
    loss_arrive = (preds[:, :num_moms_depart] - target[:, :num_moms_depart]) ** 2
    # loss_arrive = loss_arrive[loss_arrive<100000]
    moms_loss_arrive = loss_arrive.mean()

    return moms_loss_arrive


class my_Dataset_moms(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, max_lag_y=2, num_moms_y=5,
                 max_power_1_y=2, max_power_2_y=2):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.df = df

        self.num_moms_y = num_moms_y
        self.max_power_1_y = max_power_1_y
        self.max_power_2_y = max_power_2_y
        self.max_lag_y = max_lag_y

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        y = y.reshape(1, -1)

        x = x.reshape(1, -1)

        x1 = np.log(x[:, :self.num_arrival_moms])
        x2 = x[:,
             self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (self.df['mom_2'] <= self.max_power_2),
             :].index + 10]
        x3 = np.log(x[:, 135:135 + self.num_arrival_moms])
        x4 = x[:, 145 + self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (
                    self.df['mom_2'] <= self.max_power_2), :].index]
        x = np.concatenate((x1, x2, x3, x4), axis=1)
        x = torch.from_numpy(x)

        y1 = y[:, 1:self.num_moms_y]
        # y2 = y[:,df.loc[(df['lag']<=self.max_lag_y)&(df['mom_1']<=self.max_power_1_y)&(df['mom_2']<=self.max_power_2_y),:].index+10]
        # y = np.concatenate((y1,y2), axis = 1)
        y = torch.log(torch.from_numpy(y1))

        return x, y


class my_Dataset_corrs(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms=5, max_lag_y=2, num_moms_y=5,
                 max_power_1_y=2, max_power_2_y=2):
        self.data_paths = data_paths
        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.df = df

        self.num_moms_y = num_moms_y
        self.max_power_1_y = max_power_1_y
        self.max_power_2_y = max_power_2_y
        self.max_lag_y = max_lag_y

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        y = y.reshape(1, -1)

        x = x.reshape(1, -1)

        x1 = np.log(x[:, :self.num_arrival_moms])
        x2 = x[:, self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (
                    self.df['mom_2'] <= self.max_power_2), :].index + 10]
        x3 = np.log(x[:, 135:135 + self.num_arrival_moms])
        x4 = x[:, 145 + self.df.loc[(self.df['lag'] <= self.max_lag) & (self.df['mom_1'] <= self.max_power_1) & (
                    self.df['mom_2'] <= self.max_power_2), :].index]
        x = np.concatenate((x1, x2, x3, x4), axis=1)
        x = torch.from_numpy(x)

        # y1 = y[:, :self.num_moms_y]
        y2 = y[:, self.df.loc[(self.df['lag'] <= self.max_lag_y) & (self.df['mom_1'] <= self.max_power_1_y) & (
                    self.df['mom_2'] <= self.max_power_2_y), :].index + 10]
        # y = np.concatenate((y1,y2), axis = 1)
        y = torch.from_numpy(y2)

        return x, y

class MyDatasetCorrsPreloaded(Dataset):
    def __init__(self, data_paths, df, max_lag, max_power_1, max_power_2,
                 num_arrival_moms=5, max_lag_y=2, num_moms_y=5,
                 max_power_1_y=2, max_power_2_y=2):

        self.max_lag = max_lag
        self.max_power_1 = max_power_1
        self.max_power_2 = max_power_2
        self.num_arrival_moms = num_arrival_moms
        self.df = df
        self.num_moms_y = num_moms_y
        self.max_power_1_y = max_power_1_y
        self.max_power_2_y = max_power_2_y
        self.max_lag_y = max_lag_y

        mask_x = (df['lag'] <= max_lag) & (df['mom_1'] <= max_power_1) & (df['mom_2'] <= max_power_2)
        self.idx_x = df.index[mask_x].to_numpy()

        mask_y = (df['lag'] <= max_lag_y) & (df['mom_1'] <= max_power_1_y) & (df['mom_2'] <= max_power_2_y)
        self.idx_y = df.index[mask_y].to_numpy()

        X_list = []
        Y_list = []

        for path in tqdm(data_paths):
            with open(path, 'rb') as f:
                x, y = pkl.load(f)

            x = x.reshape(1, -1)
            y = y.reshape(1, -1)

            x1 = np.log(x[:, :self.num_arrival_moms])
            x2 = x[:, self.idx_x + 10]
            x3 = np.log(x[:, 135:135 + self.num_arrival_moms])
            x4 = x[:, 145 + self.idx_x]
            x_proc = np.concatenate((x1, x2, x3, x4), axis=1)

            y2 = y[:, self.idx_y + 10]

            X_list.append(torch.from_numpy(x_proc.astype(np.float32)))
            Y_list.append(torch.from_numpy(y2.astype(np.float32)))

        # concat along sample dimension
        self.X = torch.cat(X_list, dim=0)  # shape: (N, D_x)
        self.Y = torch.cat(Y_list, dim=0)  # shape: (N, D_y)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_nn_model(input_size, output_size, nn_archi):
    if nn_archi == 1:

        class Net(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 70)
                self.fc4 = nn.Linear(70, 50)
                self.fc5 = nn.Linear(50, 50)
                self.fc6 = nn.Linear(50, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = self.fc6(x)
                return x
    else:

        class Net(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 50)
                self.fc4 = nn.Linear(50, 50)
                self.fc5 = nn.Linear(50, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

    return Net(input_size, output_size)


def depart_loss_correlation(preds, target):

    weights_corr = torch.flip(torch.arange(1,target.shape[1]+1), dims=(0,))
    weights_corr = weights_corr.to(device)
    corr_loss_  =  weights_corr*torch.abs((preds[:,:]-target[:,:]))
    corr_loss = corr_loss_[corr_loss_<100000].mean()

    return corr_loss

def main():

    if sys.platform == 'linux':
        path = '/scratch/eliransc/MAP/training/merge_1'

        path_valid = '/scratch/eliransc/MAP/valid/merge_valid'
    else:

        path_valid = r'C:\Users\Eshel\workspace\data\merge_data\merge_valid'
        path = r'C:\\Users\\Eshel\\workspace\\data\\merge_data\\merge_1'

    num_arrival_moms = 5  # np.random.randint(2,10)
    max_lag = 2  # np.random.randint(1,6)
    max_power_1 = 2  # np.random.randint(1,6)
    max_power_2 = 2  # max_power_1


    df = pd.DataFrame([])

    for corr_leg in range(1, 6):
        for mom_1 in range(1, 6):
            for mom_2 in range(1, 6):
                curr_ind = df.shape[0]
                df.loc[curr_ind, 'lag'] = corr_leg
                df.loc[curr_ind, 'mom_1'] = mom_1
                df.loc[curr_ind, 'mom_2'] = mom_2
                df.loc[curr_ind, 'index'] = curr_ind + 10

        files = os.listdir(path)

    # init_ind = max_lag * max_power_1* max_power_2

    data_paths = [os.path.join(path, name) for name in files]

    batch_size = np.random.choice([64, 128])

    files_valid = os.listdir(path_valid)

    data_paths_valid = [os.path.join(path_valid, name) for name in files_valid]

    dataset = MyDatasetCorrsPreloaded(data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms)
    merged_pathes = pkl.load(open(r'C:\Users\Eshel\workspace\MAP\valid_list.pkl', 'rb'))
    dataset_valid = MyDatasetCorrsPreloaded(merged_pathes, df, max_lag, max_power_1, max_power_2, num_arrival_moms)



    # dataset_corrs = my_Dataset_corrs(data_paths, df, max_lag, max_power_1, max_power_2, num_arrival_moms)


    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=int(batch_size), shuffle=True)

    import torch
    import torch.nn as nn

    if np.random.rand() < 0.1:
        nn_archi = 1
    else:
        nn_archi = 2

    ####################################################

    ####################################################


    first_data = next(iter(loader))
    features, labels = first_data

    input_size = features.shape[-1]
    output_size = labels.shape[-1]
    output_size_corrs = 5

    from datetime import datetime

    weight_decay = 5
    curr_lr = 0.001
    EPOCHS = 120

    now = datetime.now()
    lr_second = 0.99
    lr_first = 0.75
    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    print('curr time: ', current_time)

    for indd in range(25):
        import time

        nn_archi = np.random.choice([1, 2])

        net = get_nn_model(input_size, output_size, nn_archi).to(device)



        weight_decay = 5
        curr_lr = np.random.choice([.001, .0001])
        # num_moms_corrs = 5
        now = datetime.now()
        lr_second = 0.99
        lr_first = 0.75
        current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
        print('curr time: ', current_time)


        model_num = np.random.randint(0, 1000000)

        optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                               weight_decay=(1 / 10 ** weight_decay))  # paramters is everything adjustable in model

        loss_list = []
        valid_list = []
        i = 0

        settings = (
            f"model_num_{model_num}_batch_size_{batch_size}_curr_lr_{curr_lr}_num_moms_corrs_{num_arrival_moms}_"
            f"nn_archi_{nn_archi}_max_lag_{max_lag}_max_power_1_{max_power_1}_max_power_2_{max_power_2}")

        print(settings)
        for epoch in range(EPOCHS):
            t_0 = time.time()
            for X, y in loader:
                i += 1




                # data_moms = Dataset_moms(features, labels, df, max_lag, max_power_1, max_power_2, num_arrival_moms, num_ser_moms)
                # X, y = data_moms.getitem()
                X = X.float()
                y = y.float()

                X = X.to(device)
                y = y.to(device)

                # X = X[:, 0, :]
                # y = y[:, 0, :]

                if torch.sum(torch.isinf(X)).item() == 0:
                    tt = time.time()
                    net.zero_grad()
                    output = net(X)
                    loss = depart_loss_correlation(output, y)  # 1 of two major ways to calculate loss
                    loss.backward()
                    optimizer.step()
                    net.zero_grad()

                    # print(loss)
                    if torch.isnan(loss).item():
                        break
                else:
                    pass

            loss_list.append(loss.item())
            valid_list.append(valid(loader_valid, net).item())

            init_ind = max_lag * max_power_1 * max_power_2

            df_res = check_test_corrs(loader_valid, net, init_ind, num_arrival_moms)

            df_scv, df_rhos = scv_partion(df_res)
            if sys.platform == 'linux':
                dump_path = '/scratch/eliransc/MAP/results_corrs/scv_rho_df_res' + settings + '.pkl'
                csv_file_scv = '/scratch/eliransc/MAP/results_corrs/scv_df_res' + settings + '.csv'
                csv_file_rho = '/scratch/eliransc/MAP/results_corrs/rho_df_res' + settings + '.csv'

            else:
                dump_path = r'C:\Users\Eshel\workspace\MAP\results_corrs\scv_rho_df_res' + settings + '.pkl'
                csv_file_rho = r'C:\Users\Eshel\workspace\MAP\results_corrs\rho_df_res' + settings + '.csv'
                csv_file_scv = r'C:\Users\Eshel\workspace\MAP\results_corrs\scv_df_res' + settings + '.csv'

            df_scv.to_csv(csv_file_scv, index=False)
            df_rhos.to_csv(csv_file_rho, index=False)
            pkl.dump((df_res, df_res['err1'].mean()), open(dump_path, 'wb'))

            if len(loss_list) > 3:
                if check_loss_increasing(valid_list):
                    curr_lr = curr_lr * lr_first
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                    # print(curr_lr)
                else:
                    curr_lr = curr_lr * lr_second
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                    # print(curr_lr)

            print("Epoch: {}, Training: {:.5f}, Validation : {:.5f},  , Time: {:.3f}".format(epoch, loss.item(),
                                                                                             valid_list[-1],
                                                                                             time.time() - t_0))

            if sys.platform == 'linux':

                model_path = 'eliransc/scratch/MAP/models/corr_prediction'
            else:
                model_path = r'C:\Users\Eshel\workspace\MAP\models\corr_predicition'

            file_name_model = settings  +  '.pkl'

            torch.save(net.state_dict(), os.path.join(model_path, file_name_model))


if __name__ == '__main__':

    main()




