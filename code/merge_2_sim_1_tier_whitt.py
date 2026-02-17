# imports
import simpy
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
import pandas as pd
import os
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys
sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append(r'C:\Users\Eshel\workspace\one.deep.moment')
sys.path.append(r'C:\Users\Eshel\workspace\non_renewal_queueing_system/code')
from merging import ph_to_map_renewal, create_mom_cor_vector
from utils_sample_ph import *
from utils import *
import butools
from butools.ph import *
from scipy.special import factorial
from numpy.linalg import matrix_power
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.special import factorial
is_print = False

def ser_moment_n(s, A, mom):
    '''
    ser_moment_n
    :param s:
    :param A:
    :param mom:
    :return:
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) * factorial(mom) * np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False


def compute_first_n_moments(s, A, n=3):
    '''
    compute_first_n_moments
    :param s:
    :param A:
    :param n:
    :return:
    '''
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust


is_print = False

is_print = False


class Queue_n_streams:

    def __init__(self, arrivals, services, num_arrival_streams, sim_time):

        self.env = simpy.Environment()  # initializing simpy enviroment
        self.server = simpy.Resource(self.env, capacity=1)  # Defining a resource with capacity 1
        self.end_time = sim_time  # The time simulation terminate

        self.id_current = {}  # keeping track of cusotmers id
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)
        self.arrivals = arrivals
        self.services = services
        self.num_arrival_streams = num_arrival_streams
        self.print = False

        self.global_id = 0
        for stream in range(num_arrival_streams):
            self.id_current[stream] = 0

        ### L computation
        self.num_cust_durations = np.zeros(
            500)  ## the time duration of each each state (state= number of cusotmers in the system)
        self.last_event_time = 0  # the time of the last event -
        self.num_cust_sys = 0  # keeping track of number of customers in the system

    def run(self):
        for stream in range(self.num_arrival_streams):
            self.env.process(self.customer_arrivals(stream))  ## Initializing a process
        self.env.run(until=self.end_time)  ## Running the simulaiton until self.end_time

    #########################################################
    ################# Service block #########################
    #########################################################

    def service(self, id_customer):

        arrival_time = self.env.now

        with self.server.request() as req:
            yield req

            yield self.env.timeout(self.services[self.global_id].item())
            self.global_id += 1

            # print('Customer {} complete service at {}'.format(self.global_id, self.env.now))

            ###################################
            ### Tracking number of customers ##
            ###################################

            tot_time = self.env.now - self.last_event_time  # keeping track of the last event
            self.num_cust_durations[
                self.num_cust_sys] += tot_time  # Since the number of customers in the system changes
            # print('Num cust duration', self.num_cust_durations[:5])
            if self.print:
                print(tot_time, self.num_cust_sys)

            # we compute how much time the system had this number of customers
            self.num_cust_sys -= 1  # updating number of cusotmers in the system
            self.last_event_time = self.env.now
            # print('number in the system {} last_event_time {}' .format(self.num_cust_sys , self.last_event_time))

            ###################################
            ###################################
            ###################################

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, stream_id):

        while True:
            yield self.env.timeout(self.arrivals[stream_id][self.id_current[stream_id]].item())

            self.id_current[stream_id] += 1

            id_customer = self.id_current[stream_id]

            # print('Customer {} arrived at {} from stream {}'.format(id_customer, self.env.now, stream_id))

            ###################################
            ### Tracking number of customers ##
            ###################################

            tot_time = self.env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time

            # print('Num cust duration', self.num_cust_durations[:5])

            if self.print:
                print(tot_time, self.num_cust_sys)
            self.num_cust_sys += 1
            self.last_event_time = self.env.now

            # print('number in the system {} last_event_time {}' .format(self.num_cust_sys , self.last_event_time))

            ###################################
            ###################################
            ###################################

            self.env.process(self.service(id_customer))


import numpy as np


# ---------- Helpers ----------
def poisson_arrival_trace(rate: float, T: float) -> np.ndarray:
    t = 0.0
    times = [t]
    while t < T:
        t += rng.exponential(1.0 / rate)
        times.append(t)
    return np.array(times)


def estimate_rate(event_times: np.ndarray) -> float:
    event_times = np.asarray(event_times, dtype=float)
    T = event_times[-1] - event_times[0]
    return float((len(event_times) - 1) / T)


def window_count_stats(event_times: np.ndarray, t: float, n_windows: int, rng: np.random.Generator):
    T0, T1 = event_times[0], event_times[-1]
    T = T1 - T0
    starts = T0 + rng.random(n_windows) * (T - t)
    ends = starts + t
    left = np.searchsorted(event_times, starts, side="left")
    right = np.searchsorted(event_times, ends, side="right")
    counts = right - left
    m = float(counts.mean())
    v = float(counts.var(ddof=1)) if n_windows > 1 else 0.0
    return m, v


def estimate_idc_curve(event_times: np.ndarray, t_grid: np.ndarray, n_windows: int, rng: np.random.Generator):
    lam = estimate_rate(event_times)
    idc = np.empty_like(t_grid, dtype=float)
    for k, t in enumerate(t_grid):
        m, v = window_count_stats(event_times, float(t), n_windows, rng)
        idc[k] = v / m if m > 1e-12 else np.nan
    return lam, idc


def idc_superpose(lam1, idc1, lam2, idc2):
    lam = lam1 + lam2
    w1 = lam1 / lam
    return lam, w1 * idc1 + (1.0 - w1) * idc2


def estimate_vN_from_var_slope(t_grid: np.ndarray, varN: np.ndarray, frac_tail: float = 0.4) -> float:
    n = len(t_grid)
    start = int(np.floor((1.0 - frac_tail) * n))
    start = max(0, min(n - 2, start))
    slope, _ = np.polyfit(t_grid[start:], varN[start:], 1)
    return float(slope)


def rbm_mean_wait(lam: float, ES: float, VarS: float, vN: float):
    rho = lam * ES
    if rho >= 1.0:
        raise ValueError(f"Unstable: rho={rho:.4f} >= 1")
    # work-input variance rate
    vX = lam * VarS + (ES ** 2) * vN
    EW = vX / (2.0 * (1.0 - rho))  # mean virtual waiting time / workload
    EL = lam * EW  # Little (mean number in system)
    return rho, vN, vX, EW, EL


# ---------- Stable demo ----------


if __name__ == '__main__':

    for ind in range(20):

        hyp_orig = {}
        hyp_trace = {}
        # for ind in tqdm(range(2)):
        #     if True:
        #         degree = np.random.randint(2, 4)
        #         print(degree)
        #         dat = sample(degree)
        #         A = dat[1]
        #         x_vals = np.linspace(0, 5, 70)
        #
        #         alpha = dat[0]
        #         print(compute_first_n_moments(alpha, A, 1))
        #
        #         y_vals = compute_pdf_within_range(x_vals, alpha, A)
        #         plt.figure()
        #         plt.plot(x_vals, y_vals)
        #         plt.show()
        #
        #         hyp_trace[ind] = SamplesFromPH(ml.matrix(np.array(alpha).reshape(1, -1)), ml.matrix(np.array(A)), 1000000)
        #         hyp_orig[ind] = (alpha, A)

        # path1 = r'C:\Users\Eshel\workspace\data\min_30_max_50batch_num_1673479_scv_1.pkl'
        # path2 = r'C:\Users\Eshel\workspace\data\min_10_max_30batch_num_4145285_scv_1.pkl'
        #
        # dat1 = pkl.load(open(path1, 'rb'))
        # dat2 = pkl.load(open(path2, 'rb'))
        #
        # hyp_trace[0] = dat1[-1]
        # hyp_orig[0] = (dat1[0], dat1[1])
        #
        # hyp_trace[1] = dat2[-1]
        # hyp_orig[1] = (dat2[0], dat2[1])

        ## Choose three distributionds

        path = r'C:\Users\Eshel\workspace\data\ph_examples'
        folds = os.listdir(path)
        for ind in range(3):
            choose_fold = np.random.choice(folds)
            curr_path = os.path.join(path, choose_fold)
            files = os.listdir(curr_path)
            choose_file = np.random.choice(files)
            final_path = os.path.join(curr_path, choose_file)
            dat = pkl.load(open(final_path, 'rb'))
            hyp_trace[ind] = dat[-1]
            hyp_orig[ind] = (dat[0], dat[1])
            print(final_path)

        x_vals = np.linspace(0, 5, 70)

        # for ind in range(3):
        #     y_vals = compute_pdf_within_range(x_vals, hyp_orig[ind][0], hyp_orig[ind][1])
        #     plt.figure()
        #     plt.plot(x_vals, y_vals)
        #     plt.show()





        num_streams = 2
        arrivals = {}
        for stream in range(num_streams):
            arrivals[stream] = hyp_trace[stream]

        services = hyp_trace[2]

        a = 2
        arrivals[0] = hyp_trace[0] * a
        arrivals[1] = hyp_trace[1] * (a / (a - 1))

        mom1_arrival = 1 / (1 / (arrivals[0]).mean() + 1 / (arrivals[1]).mean())
        lam = 1 / mom1_arrival

        rho = np.random.uniform(0.5, 0.9)
        meanser = rho / lam
        lam * meanser, rho

        services_norms = services * meanser
        services_norms.mean()

        ser_moms_normalized = []

        for mom in range(1, 6):
            ser_moms_normalized.append((services_norms ** mom).mean())

        arrival_a_1, arrival_A_1 = hyp_orig[0]
        arrival_a_2, arrival_A_2 = hyp_orig[1]

        arrival_A_1 = arrival_A_1 / a
        arrival_A_2 = arrival_A_2 / (a / (a - 1))

        sim_time = 748555
        queue_example1 = Queue_n_streams(arrivals, services_norms, num_streams, sim_time)
        queue_example1.run()

        L_dist = queue_example1.num_cust_durations / queue_example1.num_cust_durations.sum()



        a1 = np.cumsum(arrivals[0])
        a2 = np.cumsum(arrivals[1])

        service = services_norms

        ES = float(service.mean())
        VarS = float(service.var(ddof=1))

        # IDC grid and estimation
        t_grid = np.linspace(0.5, 80.0, 90)
        n_windows = 3000

        rng_idc = np.random.default_rng(123)
        lam1_hat, idc1 = estimate_idc_curve(a1, t_grid, n_windows, rng_idc)
        lam2_hat, idc2 = estimate_idc_curve(a2, t_grid, n_windows, rng_idc)

        lam_hat, idc_sup = idc_superpose(lam1_hat, idc1, lam2_hat, idc2)

        # Convert IDC_sup(t) to Var(N(t)) using Var(N(t)) = IDC(t) * E[N(t)] = IDC(t) * lam * t
        varN_sup = idc_sup * (lam_hat * t_grid)

        # Estimate long-run count variance rate vN from slope of Var(N(t)) vs t
        vN_hat = estimate_vN_from_var_slope(t_grid, varN_sup, frac_tail=0.4)

        # RBM mean waiting time / mean number in system
        rho, vN, vX, EW, EL = rbm_mean_wait(lam_hat, ES, VarS, vN_hat)

        # print("Estimated arrival rates:")
        # print(f"  lam1_hat={lam1_hat:.4f}, lam2_hat={lam2_hat:.4f}, lam_total={lam_hat:.4f}")
        # print("Service moments:")
        # print(f"  E[S]={ES:.6f}, Var(S)={VarS:.6f}")
        # print(f"Traffic intensity rho = lam * E[S] = {rho:.4f}  (stable if < 1)")
        # print("IDC/RBM quantities:")
        # print(f"  vN (count variance rate) = {vN:.6f}")
        # print(f"  vX (work-input variance rate) = {vX:.6f}")
        # print("RBM approximations:")
        # print(f"  E[W] ≈ {EW:.6f}")
        print(f"  E[L] ≈ {EL:.6f}")
        print((L_dist * np.arange(L_dist.shape[0])).sum(), rho)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@')









