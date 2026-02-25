# imports
import simpy
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
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

import numpy as np

# ============================================================
#  A) 4-state MAP generator (Exp-by-state Markov-renewal MAP)
#
#  Family:
#    D0 = -diag(lam_1,...,lam_n)
#    D1 = diag(lam_1,...,lam_n) @ P
#  where P is row-stochastic (post-arrival transition matrix).
#
#  Inter-arrival time conditional on state i is Exp(lam_i).
#  Lag-1 correlation is induced by P.
#
#  Mean is scaled to 1 by multiplying all rates by c = mean.
#  (SCV and rho1 are unchanged by this scaling.)
# ============================================================

def stationary_dist_P(P: np.ndarray) -> np.ndarray:
    """Stationary distribution alpha (row vector entries) for row-stochastic P: alpha = alpha P."""
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)

def metrics_exp_by_state_MAP(lams: np.ndarray, P: np.ndarray):
    """
    Closed-form mean, SCV, and lag-1 autocorrelation rho1 for Exp-by-state MAP.
    """
    lams = np.asarray(lams, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)
    n = len(lams)
    assert P.shape == (n, n)
    assert np.all(lams > 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10)

    alpha = stationary_dist_P(P)  # post-arrival stationary distribution

    m = 1.0 / lams                  # E[T | state=i]
    m2 = 2.0 / (lams ** 2)          # E[T^2 | state=i]

    mean = float(alpha @ m)
    ET2 = float(alpha @ m2)
    var = ET2 - mean * mean
    scv = var / (mean * mean)

    next_mean_given_i = P @ m
    ET1T2 = float(alpha @ (m * next_mean_given_i))
    rho1 = (ET1T2 - mean * mean) / var

    return mean, scv, rho1

def build_exp_by_state_MAP(lams: np.ndarray, P: np.ndarray):
    """Return (D0,D1) for the Exp-by-state MAP."""
    lams = np.asarray(lams, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)
    D0 = -np.diag(lams)
    D1 = np.diag(lams) @ P
    return D0, D1

def scale_map_mean_to_1(lams: np.ndarray, P: np.ndarray):
    """Scale all MAP rates so mean inter-arrival time becomes 1."""
    mean, scv, rho1 = metrics_exp_by_state_MAP(lams, P)
    c = mean
    lams2 = lams * c
    mean2, scv2, rho12 = metrics_exp_by_state_MAP(lams2, P)
    # mean2 should be 1 (up to numerical eps)
    return lams2, P, (mean2, scv2, rho12)

def make_grouped_transition_matrix(n=4, s_cross=0.5, stick=0.0, rng=None):
    """
    Create a structured P that can induce negative or positive correlation depending on s_cross / stick.
      - We split states into two groups: fast (0..n/2-1) and slow (n/2..n-1)
      - s_cross controls switching between groups:
           s_cross > 0.5 -> anti-persistent -> tends to negative rho1
           s_cross < 0.5 -> persistent (within group) -> tends to positive rho1
      - stick adds probability of staying in the same state (stronger positive corr)
    """
    if rng is None:
        rng = np.random.default_rng()

    assert n % 2 == 0, "Use even n for the grouped construction."
    g = n // 2
    fast = list(range(g))
    slow = list(range(g, n))

    P = np.zeros((n, n), dtype=float)

    # Distribute probability mass uniformly within destination group
    def uniform_over(indices):
        w = np.zeros(n, dtype=float)
        w[indices] = 1.0 / len(indices)
        return w

    w_fast = uniform_over(fast)
    w_slow = uniform_over(slow)

    for i in range(n):
        # base: mix within group vs across group
        if i in fast:
            base = (1 - s_cross) * w_fast + s_cross * w_slow
        else:
            base = (1 - s_cross) * w_slow + s_cross * w_fast

        # add "stickiness" to same state
        base = (1 - stick) * base
        base[i] += stick

        # tiny jitter to avoid exact symmetry / degeneracy
        base += rng.uniform(0.0, 1e-6, size=n)
        base /= base.sum()
        P[i, :] = base

    return P

def sample_lams_for_scv_range(rng, n=4, scv_target=None):
    """
    Heuristic sampling of rates that tends to produce a broad SCV range.
    If scv_target is given, it biases the disparity accordingly (still approximate).
    """
    # We create 2 fast and 2 slow rates by default (n=4).
    # Larger disparity => larger SCV.
    if n != 4:
        # generic: log-uniform rates
        lams = np.exp(rng.uniform(np.log(0.1), np.log(200.0), size=n))
        return lams

    # Choose a base "fast" scale
    lam_fast = float(np.exp(rng.uniform(np.log(2.0), np.log(200.0))))

    # Choose disparity according to target SCV (rough bias, not exact)
    if scv_target is None:
        ratio = float(np.exp(rng.uniform(np.log(1.0), np.log(2000.0))))
    else:
        # bigger SCV -> bigger ratio
        # map scv_target in [0.1, 8] to ratio range roughly [1, 2000]
        x = (np.clip(scv_target, 0.1, 8.0) - 0.1) / (8.0 - 0.1)
        log_ratio = np.log(1.0) * (1 - x) + np.log(2000.0) * x
        ratio = float(np.exp(rng.uniform(max(np.log(1.0), log_ratio - 1.0),
                                         min(np.log(2000.0), log_ratio + 1.0))))

    lam_slow = lam_fast / ratio

    # add mild variability within fast/slow groups
    lam_fast2 = lam_fast * float(np.exp(rng.uniform(-0.5, 0.5)))
    lam_slow2 = lam_slow * float(np.exp(rng.uniform(-0.5, 0.5)))

    lams = np.array([lam_fast, lam_fast2, lam_slow, lam_slow2], dtype=float)
    lams = np.clip(lams, 1e-6, None)
    return lams

def generate_many_MAPs(
    num_maps=200,
    scv_min=0.1, scv_max=8.0,
    rho_min=-0.2, rho_max=0.2,
    n=4,
    seed=1,
    max_tries=2_000_000,
):
    """
    Rejection-sample a wide range of 4-state MAPs with mean=1, SCV in [scv_min, scv_max],
    and lag-1 autocorrelation in [rho_min, rho_max].

    Returns a list of dicts:
      {"D0":..., "D1":..., "mean":..., "scv":..., "rho1":..., "lams":..., "P":...}
    """
    rng = np.random.default_rng(seed)
    out = []
    tries = 0

    while len(out) < num_maps and tries < max_tries:
        tries += 1

        # sample a target to encourage coverage
        scv_t = float(rng.uniform(scv_min, scv_max))
        rho_t = float(rng.uniform(rho_min, rho_max))

        lams = sample_lams_for_scv_range(rng, n=n, scv_target=scv_t)

        # choose P parameters to steer rho sign/scale
        # s_cross > 0.5 tends negative; < 0.5 tends positive
        s_cross = 0.5 + 0.35 * np.tanh(-5.0 * rho_t)  # heuristic mapping
        s_cross = float(np.clip(s_cross + rng.normal(0, 0.05), 0.0, 1.0))

        # stickiness can boost positive correlation
        stick = float(np.clip(0.6 * max(0.0, rho_t) + rng.uniform(0.0, 0.1), 0.0, 0.85))

        P = make_grouped_transition_matrix(n=n, s_cross=s_cross, stick=stick, rng=rng)

        # scale mean to 1
        lams2, P2, (mean, scv, rho1) = scale_map_mean_to_1(lams, P)

        if (scv_min <= scv <= scv_max) and (rho_min <= rho1 <= rho_max):
            D0, D1 = build_exp_by_state_MAP(lams2, P2)
            out.append({
                "D0": D0,
                "D1": D1,
                "mean": mean,
                "scv": scv,
                "rho1": rho1,
                "lams": lams2,
                "P": P2,
            })

    if len(out) < num_maps:
        print(f"Warning: only generated {len(out)} MAPs in {tries} tries. "
              f"Try increasing max_tries or loosening ranges.")

    return out


# ============================================================
#  B) PH generator with mean=1 and SCV in [0.1, 8]
#
#  For SCV <= 1:
#    Hyper-Erlang mixture of two Erlangs with the SAME mean 1:
#      Erlang(k, rate=k) has mean 1 and SCV = 1/k
#    Mixing two such Erlangs (with same mean) yields SCV that is
#    a convex combination of 1/k1 and 1/k2 (exact control).
#
#  For SCV > 1:
#    2-phase Hyperexponential with equal mixing p=0.5:
#      choose m1, m2 such that:
#         (m1 + m2)/2 = 1
#         SCV = (m1^2 + m2^2) - 1
#      closed-form:
#         m1 = 1 + 0.5*sqrt(2*(SCV-1)),  m2 = 2 - m1
#      rates mu_i = 1/m_i, alpha=[0.5,0.5], T=diag(-mu1,-mu2)
# ============================================================

def ph_mean_scv(alpha, T):
    """Compute mean and SCV of PH(alpha,T) via moments."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=float)
    s = T.shape[0]
    e = np.ones(s)

    invT = np.linalg.solve(-T, np.eye(s))
    m1 = float(alpha @ (invT @ e))
    m2 = float(2.0 * alpha @ (invT @ invT @ e))
    var = m2 - m1 * m1
    scv = var / (m1 * m1)
    return m1, scv

def make_ph_hyperexp2_mean1(target_scv):
    """2-phase hyperexponential (p=0.5) with mean=1 and SCV=target_scv (>1)."""
    c2 = float(target_scv)
    assert c2 > 1.0
    d = 0.5 * np.sqrt(2.0 * (c2 - 1.0))
    m1 = 1.0 + d
    m2 = 1.0 - d
    # ensure positive
    m2 = max(m2, 1e-8)

    mu1 = 1.0 / m1
    mu2 = 1.0 / m2

    alpha = np.array([0.5, 0.5], dtype=float)
    T = np.array([[-mu1, 0.0],
                  [0.0, -mu2]], dtype=float)
    return alpha, T

def make_ph_hypererlang_mean1(target_scv, k_max=200):
    """
    Mixture of two Erlangs with mean 1 to match SCV in (0,1].
    Erlang(k, rate=k) has mean 1 and SCV = 1/k.
    We pick k1,k2 such that 1/k brackets target_scv and mix linearly.
    """
    c2 = float(target_scv)
    assert 0.0 < c2 <= 1.0 + 1e-12

    # choose k_low <= 1/c2 <= k_high; then SCV between 1/k_low and 1/k_high
    k_star = 1.0 / c2
    k1 = int(np.floor(k_star))
    k2 = int(np.ceil(k_star))
    k1 = max(1, min(k1, k_max))
    k2 = max(1, min(k2, k_max))

    if k1 == k2:
        # exact Erlang
        k = k1
        # Erlang(k, rate=k): Coxian chain of k phases
        alpha = np.zeros(k)
        alpha[0] = 1.0
        T = np.zeros((k, k))
        for i in range(k):
            T[i, i] = -k
            if i < k - 1:
                T[i, i + 1] = k
        return alpha, T

    scv1 = 1.0 / k1
    scv2 = 1.0 / k2

    # convex combination to match target SCV:
    # c2 = p*scv1 + (1-p)*scv2  => p = (c2 - scv2)/(scv1 - scv2)
    p = (c2 - scv2) / (scv1 - scv2)
    p = float(np.clip(p, 0.0, 1.0))

    # Hyper-Erlang as PH: block-diagonal of two Erlangs
    # Component 1: Erlang(k1, rate=k1)
    # Component 2: Erlang(k2, rate=k2)
    s = k1 + k2
    alpha = np.zeros(s, dtype=float)
    alpha[0] = p
    alpha[k1] = 1.0 - p

    T = np.zeros((s, s), dtype=float)

    # block 1
    for i in range(k1):
        T[i, i] = -k1
        if i < k1 - 1:
            T[i, i + 1] = k1

    # block 2
    off = k1
    for i in range(k2):
        T[off + i, off + i] = -k2
        if i < k2 - 1:
            T[off + i, off + i + 1] = k2

    return alpha, T

def generate_many_PHs(
    num_ph=200,
    scv_min=0.1, scv_max=8.0,
    seed=1
):
    """
    Generate PH distributions with mean=1 and SCV in [scv_min, scv_max].
    Returns list of dicts:
      {"alpha":..., "T":..., "mean":..., "scv":...}
    """
    rng = np.random.default_rng(seed)
    out = []

    for _ in range(num_ph):
        c2 = float(rng.uniform(scv_min, scv_max))
        if c2 <= 1.0:
            alpha, T = make_ph_hypererlang_mean1(c2)
        else:
            alpha, T = make_ph_hyperexp2_mean1(c2)

        mean, scv = ph_mean_scv(alpha, T)
        out.append({"alpha": alpha, "T": T, "mean": mean, "scv": scv})

    return out

import numpy as np

# ---------------------------
# Moment check (optional)
# ---------------------------
def ph_mean_scv(alpha, T):
    alpha = np.asarray(alpha, float).reshape(-1)
    T = np.asarray(T, float)
    s = T.shape[0]
    e = np.ones(s)

    invT = np.linalg.solve(-T, np.eye(s))
    m1 = float(alpha @ (invT @ e))
    m2 = float(2.0 * alpha @ (invT @ invT @ e))
    var = m2 - m1 * m1
    scv = var / (m1 * m1)
    return m1, scv


# ============================================================
# PH for SCV > 1: 2-phase Hyperexponential with flexible mixing p
#
# Mean constraint:
#   p*m1 + (1-p)*m2 = 1
# Second moment constraint (Exp component i has E[X^2]=2*m_i^2):
#   E[X^2] = 2*(p*m1^2 + (1-p)*m2^2) = 1 + SCV    (since mean=1 => Var=SCV)
#
# Choose p (small p => can reach huge SCV), solve for m1>0 and m2>0.
# ============================================================

def make_ph_hyperexp2_mean1(target_scv, p=None):
    """
    Construct 2-phase hyperexponential PH with mean=1 and SCV=target_scv (>1).
    Returns (alpha, T).
    """
    c2 = float(target_scv)
    if c2 <= 1.0:
        raise ValueError("Hyperexponential is for SCV > 1.")

    # Pick a mixing probability if not provided.
    # For c2 up to 8, p around 0.1-0.3 is typically fine; for larger c2, p smaller helps.
    if p is None:
        # heuristic: p decreases as SCV increases, but keep it away from 0
        p = min(0.45, max(0.02, 1.0 / (c2 + 1.0)))
        # for SCV around 8.7 => p ~ 0.103; for SCV=8 => p~0.111; for SCV=2 => p~0.333

    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1).")

    # Let m1 be unknown (mean of Exp component 1), then:
    # m2 = (1 - p*m1)/(1-p)
    # Enforce E[X^2] = 1 + c2:
    # 2*(p*m1^2 + (1-p)*m2^2) = 1 + c2
    #
    # This becomes a quadratic in m1. We'll solve via np.roots.

    # Expand:
    # 2*p*m1^2 + 2*(1 - p*m1)^2/(1-p) = 1 + c2
    A = 2.0 * p + 2.0 * (p**2) / (1.0 - p)
    B = -4.0 * p / (1.0 - p)
    C = 2.0 / (1.0 - p) - (1.0 + c2)

    roots = np.roots([A, B, C])

    candidates = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            m1 = float(r.real)
            if m1 > 0:
                m2 = (1.0 - p * m1) / (1.0 - p)
                if m2 > 0:
                    candidates.append((m1, m2))

    if not candidates:
        raise RuntimeError("Failed to find positive (m1,m2). Try a smaller p (e.g. p=0.05).")

    # Pick a numerically stable candidate (avoid extremely tiny mean unless necessary)
    candidates.sort(key=lambda x: x[0])
    m1, m2 = candidates[0]  # usually gives "smaller m1, larger m2" which creates high SCV

    mu1 = 1.0 / m1
    mu2 = 1.0 / m2

    alpha = np.array([p, 1.0 - p], dtype=float)
    T = np.array([[-mu1, 0.0],
                  [0.0, -mu2]], dtype=float)
    return alpha, T


# ============================================================
# PH for SCV <= 1: Hyper-Erlang mixture with mean=1
# Erlang(k, rate=k): mean 1, SCV=1/k
# Mixture of two such Erlangs (same mean) gives SCV between them.
# ============================================================

def make_ph_hypererlang_mean1(target_scv, k_max=400):
    c2 = float(target_scv)
    if not (0.0 < c2 <= 1.0 + 1e-12):
        raise ValueError("Hyper-Erlang here is for SCV in (0,1].")

    k_star = 1.0 / c2
    k1 = int(np.floor(k_star))
    k2 = int(np.ceil(k_star))
    k1 = max(1, min(k1, k_max))
    k2 = max(1, min(k2, k_max))

    if k1 == k2:
        k = k1
        alpha = np.zeros(k)
        alpha[0] = 1.0
        T = np.zeros((k, k))
        for i in range(k):
            T[i, i] = -k
            if i < k - 1:
                T[i, i + 1] = k
        return alpha, T

    scv1 = 1.0 / k1
    scv2 = 1.0 / k2
    p = (c2 - scv2) / (scv1 - scv2)
    p = float(np.clip(p, 0.0, 1.0))

    s = k1 + k2
    alpha = np.zeros(s, dtype=float)
    alpha[0] = p
    alpha[k1] = 1.0 - p

    T = np.zeros((s, s), dtype=float)

    # Erlang block 1
    for i in range(k1):
        T[i, i] = -k1
        if i < k1 - 1:
            T[i, i + 1] = k1

    # Erlang block 2
    off = k1
    for i in range(k2):
        T[off + i, off + i] = -k2
        if i < k2 - 1:
            T[off + i, off + i + 1] = k2

    return alpha, T


def make_ph_mean1_with_scv(target_scv):
    """
    Unified PH constructor with mean=1 and desired SCV in (0, inf).
    Uses:
      - Hyper-Erlang for SCV <= 1
      - HyperExp-2 (flexible p) for SCV > 1 (can reach SCV >> 3)
    """
    c2 = float(target_scv)
    if c2 <= 1.0:
        return make_ph_hypererlang_mean1(c2)
    else:
        return make_ph_hyperexp2_mean1(c2)



def superpose_map(D0a: np.ndarray, D1a: np.ndarray,
                  D0b: np.ndarray, D1b: np.ndarray,
                  check: bool = True):
    """
    Superposition of two independent MAPs (D0a,D1a) and (D0b,D1b).
    Returns (D0s, D1s) of size (m*n) x (m*n).

    Convention: Q = D0 + D1 is the generator of the background CTMC.
    """
    D0a = np.asarray(D0a, dtype=float)
    D1a = np.asarray(D1a, dtype=float)
    D0b = np.asarray(D0b, dtype=float)
    D1b = np.asarray(D1b, dtype=float)

    if D0a.shape != D1a.shape or D0a.ndim != 2 or D0a.shape[0] != D0a.shape[1]:
        raise ValueError("MAP A: D0a and D1a must be square and same shape.")
    if D0b.shape != D1b.shape or D0b.ndim != 2 or D0b.shape[0] != D0b.shape[1]:
        raise ValueError("MAP B: D0b and D1b must be square and same shape.")

    m = D0a.shape[0]
    n = D0b.shape[0]
    Im = np.eye(m)
    In = np.eye(n)

    D0s = np.kron(D0a, In) + np.kron(Im, D0b)
    D1s = np.kron(D1a, In) + np.kron(Im, D1b)

    if check:
        Qa = D0a + D1a
        Qb = D0b + D1b
        # Basic MAP checks: row sums ~ 0 for Q
        if not np.allclose(Qa.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("MAP A check failed: rows of (D0a + D1a) must sum to 0.")
        if not np.allclose(Qb.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("MAP B check failed: rows of (D0b + D1b) must sum to 0.")
        Qs = D0s + D1s
        if not np.allclose(Qs.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("Superposed MAP check failed: rows of (D0s + D1s) must sum to 0.")

    return D0s, D1s


def superpose_initial(alpha_a: np.ndarray, alpha_b: np.ndarray):
    """
    If you also have initial distributions alpha_a (1xm) and alpha_b (1xn),
    the natural product initial distribution is kron(alpha_a, alpha_b).
    """
    aa = np.asarray(alpha_a, dtype=float).reshape(1, -1)
    ab = np.asarray(alpha_b, dtype=float).reshape(1, -1)
    return np.kron(aa, ab)




def stationary_pi(Q):
    n = Q.shape[0]
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def map_mean_scv_rho1(D0, D1):
    n = D0.shape[0]
    Q = D0 + D1
    pi = stationary_pi(Q)

    ones = np.ones(n)
    lam = float(pi @ (D1 @ ones))      # arrival rate
    alpha = (pi @ D1) / lam            # post-arrival state distribution

    M = np.linalg.inv(-D0)

    mean = float(alpha @ (M @ ones))
    m2 = float(2 * alpha @ (M @ M @ ones))
    var = m2 - mean**2
    scv = var / mean**2

    mvec = M @ ones
    ET1T2 = float(alpha @ (M @ M @ (D1 @ mvec)))
    rho1 = (ET1T2 - mean**2) / var

    return mean, scv, rho1


def scale_mean_to_one(D0, D1):
    """
    Scale time so mean interarrival = 1.
    """
    mean, _, _ = map_mean_scv_rho1(D0, D1)
    c = mean  # multiply rates by mean
    return D0 * c, D1 * c


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


import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve


def stationary_dist_ctmc(Q: np.ndarray) -> np.ndarray:
    """
    Stationary distribution pi (as a row vector entries) of CTMC generator Q:
      pi Q = 0, pi 1 = 1
    Returns pi as a 1D array of length n (interpreted as row).
    """
    n = Q.shape[0]
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = solve(A, b)
    return pi  # 1D


def map_rate(D0: np.ndarray, D1: np.ndarray) -> float:
    """Arrival rate lambda for MAP(D0,D1)."""
    Q = D0 + D1
    pi = stationary_dist_ctmc(Q)
    ones = np.ones(Q.shape[0])
    return float(pi @ (D1 @ ones))


def ph_mean(alpha: np.ndarray, T: np.ndarray) -> float:
    """Mean of PH(alpha,T). alpha row vector length s."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    ones = np.ones(T.shape[0])
    return float(alpha @ solve(-T, ones))


def compute_R(A_up: np.ndarray, A_same: np.ndarray, A_down: np.ndarray,
              max_iter: int = 20000, tol: float = 1e-12) -> np.ndarray:
    """
    Solve A_up + R A_same + R^2 A_down = 0 for minimal nonnegative R
    via functional iteration:
      R_{k+1} = -A_up (A_same + R_k A_down)^{-1}
    """
    n = A_same.shape[0]
    R = np.zeros((n, n))
    for k in range(max_iter):
        M = A_same + R @ A_down
        R_next = solve(M.T, (-A_up).T).T
        err = norm(R_next - R, ord=np.inf)
        R = R_next
        if err < tol:
            return R
    raise RuntimeError(f"R iteration did not converge in {max_iter} iterations (last err={err})")


def map_ph_1_steady_state_with_marginal(
        D0: np.ndarray, D1: np.ndarray,
        alpha: np.ndarray, T: np.ndarray,
        N_max: int = 200,
        r_tol: float = 1e-12
) -> dict:
    """
    Compute steady state for MAP/PH/1 and return marginal P(N=n) for n=0..N_max.

    Levels:
      - Level 0: MAP background only (size m)
      - Level n>=1: MAP background × PH service phase (size m*s)

    Returns:
      pN: length N_max+1 with P(N=n)
      tail: P(N > N_max)
      plus pi0, pi1, R, rho, etc.
    """
    D0 = np.asarray(D0, dtype=float)
    D1 = np.asarray(D1, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=float)

    m = D0.shape[0]
    s = T.shape[0]
    ms = m * s

    # --- stability check
    lam = map_rate(D0, D1)
    ES = ph_mean(alpha, T)
    rho = lam * ES
    if rho >= 1.0 - 1e-12:
        raise ValueError(f"Unstable or nearly unstable: rho = {rho} (need < 1).")

    # --- PH completion vector
    ones_s = np.ones(s)
    t = -T @ ones_s  # (s,)

    I_m = np.eye(m)
    I_s = np.eye(s)

    # QBD blocks for levels n>=1 (size ms x ms)
    A_same = np.kron(D0, I_s) + np.kron(I_m, T)
    A_up = np.kron(D1, I_s)
    A_down = np.kron(I_m, np.outer(t, alpha))  # (ms x ms)

    # boundary blocks
    B01 = np.kron(D1, alpha.reshape(1, -1))  # (m x ms)
    B10 = np.kron(I_m, t.reshape(-1, 1))  # (ms x m)

    # --- compute R
    R = compute_R(A_up, A_same, A_down, tol=r_tol)

    # --- solve for pi0 (m) and pi1 (ms)
    M_top = np.hstack([D0.T, B10.T])  # m x (m+ms)
    M_mid = np.hstack([B01.T, (A_same + R @ A_down).T])  # ms x (m+ms)
    M = np.vstack([M_top, M_mid])  # (m+ms) x (m+ms)

    # normalization: pi0*1 + pi1*(I-R)^{-1}*1 = 1
    ones_m = np.ones(m)
    ones_ms = np.ones(ms)
    w = solve(np.eye(ms) - R, ones_ms)  # (I-R)^{-1} 1
    norm_row = np.concatenate([ones_m, w])

    b = np.zeros(m + ms)
    M[-1, :] = norm_row
    b[-1] = 1.0

    u = solve(M, b)
    pi0 = u[:m]
    pi1 = u[m:]

    # --- marginal P(N=n) for n=0..N_max
    pN = np.zeros(N_max + 1, dtype=float)
    pN[0] = float(pi0.sum())

    # iterate pi_n = pi1 R^{n-1} for n>=1
    cur = pi1.copy()
    for n in range(1, N_max + 1):
        pN[n] = float(cur.sum())
        cur = cur @ R

    # --- tail mass P(N > N_max)
    # Total mass at levels >=1 is pi1*(I-R)^{-1}*1
    total_ge1 = float(pi1 @ w)
    mass_1_to_N = float(pN[1:].sum())
    tail = max(0.0, total_ge1 - mass_1_to_N)

    return {
        "rho": rho,
        "lambda": lam,
        "ES": ES,
        "R": R,
        "pi0": pi0,
        "pi1": pi1,
        "pN_0_to_Nmax": pN,  # length N_max+1
        "tail_gt_Nmax": tail,  # P(N > N_max)
        "check_total": float(pN.sum() + tail)  # should be ~1
    }

def choose_scv_larger_3(maps):

    for ind in range(len(maps)):

        ind_rand = np.random.randint(0, len(maps))

        if maps[ind_rand]['scv']>3:
            return ind_rand

    return 0

def choose_scv_lower_3(maps):

    for ind in range(len(maps)):

        ind_rand = np.random.randint(0, len(maps))

        if maps[ind_rand]['scv']<3:
            return ind_rand

    return 0


def choose_scv_larger_3_rho_smaller_0(maps):

    for ind in range(len(maps)):

        ind_rand = np.random.randint(0, len(maps))

        if (maps[ind_rand]['scv']>3)&(maps[ind_rand]['rho1']<0):
            return ind_rand

    return 0


def merge_two_random_stream():
    hyp_orig = {}
    hyp_trace = {}

    maps = generate_many_MAPs(
        num_maps=1500,
        scv_min=0.1, scv_max=8.0,
        rho_min=-0.2, rho_max=0.2,
        n=4,
        max_tries=1500000
    )

    # Generate PHs
    phs = generate_many_PHs(
        num_ph=100,
        scv_min=0.4, scv_max=8.0
    )

    ind_ph = np.random.randint(0, 100)
    alpha, T = phs[ind_ph]['alpha'], phs[ind_ph]['T']
    T = T * phs[ind_ph]['mean']
    rho = np.random.uniform(0.65, 0.85)
    T = T / rho
    res_ph = compute_first_n_moments(alpha, T, 10)

    moms_ser = np.log(np.array(res_ph).flatten())

    a = np.random.uniform(1, 5)
    if np.random.rand() < 0.5:
        ind_map1 = choose_scv_lower_3(maps)  # np.random.randint(0, 500)
        ind_map2 = choose_scv_larger_3(maps)  # np.random.randint(0, 500)
    else:
        ind_map2 = choose_scv_lower_3(maps)  # np.random.randint(0, 500)
        ind_map1 = choose_scv_larger_3(maps)

    DO_merged, D1_merged = superpose_map((1 / (a - 1)) * maps[ind_map1]['D0'].copy(),
                                         (1 / (a - 1)) * maps[ind_map1]['D1'].copy(),
                                         maps[ind_map2]['D0'].copy(), maps[ind_map2]['D1'].copy())
    mom1 = create_mom_cor_vector(DO_merged, D1_merged)[0]
    DO_merged, D1_merged = DO_merged * mom1, D1_merged * mom1
    res_map = create_mom_cor_vector(DO_merged, D1_merged)
    res_map[:10] = np.log(res_map[:10])
    final_inp = np.concatenate((res_map, moms_ser))
    res_steady = map_ph_1_steady_state_with_marginal(DO_merged, D1_merged, alpha, T, N_max=200)
    pN = res_steady["pN_0_to_Nmax"]

    map_trace, map_vector = pkl.load(open(r'C:\Users\Eshel\workspace\MAP\map_for_queueing.pkl', 'rb'))
    mom_cors = {}
    choen_inds = []
    # for ind in range(2):
    #     if np.random.rand() < 0.25:
    #         ind_c = 44 #np.random.choice(np.arange(len(map_vector.keys())))
    #     else:
    #         ind_c = 44 # np.random.randint(20, len(map_vector.keys()))  ##np.random.choice(list(map_trace.keys()))
    #     choen_inds.append(ind_c)
    #     hyp_trace[ind] = map_trace[ind_c]
    #     mom_cors[ind] = map_vector[ind_c]

    x_vals = np.linspace(0, 5, 70)
    hyp_trace[0] = SamplesFromMAP(ml.matrix(maps[ind_map1]['D0'].copy()), ml.matrix(maps[ind_map1]['D1'].copy()),
                                  1000000)
    hyp_trace[1] = SamplesFromMAP(ml.matrix(maps[ind_map2]['D0'].copy()), ml.matrix(maps[ind_map2]['D1'].copy()),
                                  1000000)

    map_vector[0] = create_mom_cor_vector(maps[ind_map1]['D0'].copy(), maps[ind_map1]['D1'].copy())
    map_vector[1] = create_mom_cor_vector(maps[ind_map2]['D0'].copy(), maps[ind_map2]['D1'].copy())

    mom_cors[0] = map_vector[0].copy()
    mom_cors[1] = map_vector[1].copy()

    mom_1_a = map_vector[0][0]
    mom_2_a = map_vector[0][1]
    SCV_1 = (mom_2_a - mom_1_a ** 2) / mom_1_a ** 2
    mom_1_b = map_vector[1][0]
    mom_2_b = map_vector[1][1]
    SCV_2 = (mom_2_b - mom_1_b ** 2) / mom_1_b ** 2
    rho_a = map_vector[0][10]
    rho_b = map_vector[1][10]
    print(SCV_1, SCV_2, rho_a, rho_b)

    # for ind in range(3):
    #     y_vals = compute_pdf_within_range(x_vals, hyp_orig[ind][0], hyp_orig[ind][1])
    #     plt.figure()
    #     plt.plot(x_vals, y_vals)
    #     plt.show()

    chosen_pathes = []
    # path = r'C:\Users\Eshel\workspace\data\ph_examples'
    # folds = os.listdir(path)

    # choose_fold = '1' #np.random.choice(folds)
    #
    # curr_path = os.path.join(path, choose_fold)
    # files = os.listdir(curr_path)
    # choose_file = np.random.choice(files)
    # final_path = os.path.join(curr_path, choose_file)
    # print(final_path)
    # dat = pkl.load(open(final_path, 'rb'))
    hyp_trace[2] = SamplesFromPH(ml.matrix(alpha), ml.matrix(T), 1000000)
    hyp_orig[2] = (alpha, T)

    res_ph = np.array(compute_first_n_moments(alpha, T, 10)).flatten()
    # print(final_path)
    # chosen_pathes.append(final_path)
    SCV_ser = (res_ph[1] - res_ph[0] ** 2) / res_ph[0] ** 2  ##dat[2][1]-1

    num_streams = 2
    arrivals = {}
    for stream in range(num_streams):
        arrivals[stream] = hyp_trace[stream]

    services = hyp_trace[2]

    # a = np.random.uniform(1,5)
    arrivals[0] = hyp_trace[0] * a
    arrivals[1] = hyp_trace[1] * (a / (a - 1))

    mom1_arrival = 1 / (1 / (arrivals[0]).mean() + 1 / (arrivals[1]).mean())
    lam = 1 / mom1_arrival

    # rho =  0.75 #np.random.uniform(0.4, 0.9)
    meanser = rho / lam
    lam * meanser, rho

    # services_norms = services * meanser
    # services_norms.mean()
    #
    # ser_moms_normalized = []
    #
    # for mom in range(1, 6):
    #     ser_moms_normalized.append((services_norms ** mom).mean())

    #
    # sim_time = 1144855
    # queue_example1 = Queue_n_streams(arrivals, services_norms, num_streams, sim_time)
    # queue_example1.run()
    #
    # L_dist = queue_example1.num_cust_durations / queue_example1.num_cust_durations.sum()

    L_dist = pN / pN.sum()
    import torch
    ser_torch = torch.tensor(moms_ser[:5], dtype=torch.float32)  # torch.tensor(np.array(ser_moms_normalized), dtype=torch.float32)
    ser_torch = ser_torch.to(device)
    ser_torch = ser_torch.reshape(1, -1)
    # ser_torch = torch.log(ser_torch)

    # step 2
    scale = a - 1
    resa = mom_cors[0].copy()
    resb = mom_cors[1].copy()
    print(resb[0], resa[0])
    for mom in range(1, 11):
        resa[mom - 1] = resa[mom - 1] * (scale ** mom)

    # step 2 cont'
    max_lag = 2
    max_power_1 = 2
    max_power_2 = 2
    num_arrive_moms = 5
    df = pd.DataFrame([])

    for corr_leg in range(1, 6):
        for mom_1 in range(1, 6):
            for mom_2 in range(1, 6):
                curr_ind = df.shape[0]
                df.loc[curr_ind, 'lag'] = corr_leg
                df.loc[curr_ind, 'mom_1'] = mom_1
                df.loc[curr_ind, 'mom_2'] = mom_2
                df.loc[curr_ind, 'index'] = curr_ind + 10

    inp1_moms = resa[:num_arrive_moms]
    inp2_moms = resb[:num_arrive_moms]

    inp1_corrs = resa[
        df.loc[(df['lag'] <= max_lag) & (df['mom_1'] <= max_power_1) & (df['mom_2'] <= max_power_2), :].index + 10]
    inp2_corrs = resb[
        df.loc[(df['lag'] <= max_lag) & (df['mom_1'] <= max_power_1) & (df['mom_2'] <= max_power_2), :].index + 10]

    tot_inp = np.concatenate((np.log(inp1_moms), inp1_corrs, np.log(inp2_moms), inp2_corrs))
    tot_inp = torch.tensor(tot_inp, dtype=torch.float32)

    # step 3
    import torch.nn.functional as F
    import torch
    import torch.nn as nn

    class Net_moms(nn.Module):

        def __init__(self, input_size, output_size):
            super().__init__()

            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 50)
            self.fc4 = nn.Linear(50, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    class Net_moms(nn.Module):

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

    class Net_corrs(nn.Module):

        def __init__(self, input_size, output_size):
            super().__init__()

            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 50)
            self.fc4 = nn.Linear(50, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    input_size_moms, output_size_moms = 26, 4

    net_moms = Net_moms(input_size_moms, output_size_moms).to(device)

    input_size_corrs, output_size_corrs = 26, 8

    # net_corrs = Net_corrs(input_size_corrs, output_size_corrs).to(device)

    model_path = r'C:\Users\Eshel\workspace\MAP\models\moment_prediction'  # r'C:\Users\Eshel\workspace\MAP\models'
    file_name_model_corrs = 'merge_corrs_1.pkl'
    file_name_model_moms = 'model_num_97370_batch_size_64_curr_lr_0.001_num_moms_corrs_5_nn_archi_1_max_lag_2_max_power_1_2_max_power_2_2.pkl'  # 'merge_moments_1.pkl' #
    model_path_mom = r'C:\Users\Eshel\workspace\MAP\models'
    # file_name_model_moms = 'merge_moments_1.pkl'

    net_moms.load_state_dict(torch.load(os.path.join(model_path, file_name_model_moms)))
    net_moms.to(device)
    # net_moms.load_state_dict(torch.load(os.path.join(model_path, file_name_model_moms)))
    # net_corrs.load_state_dict(torch.load(os.path.join(model_path, file_name_model_corrs)))

    input_size_corrs, output_size_corrs = 26, 8
    nn_archi = 1
    net_corrs = get_nn_model(input_size_corrs, output_size_corrs,
                             nn_archi)  # Net_corrs(input_size_corrs, output_size_corrs) ##
    model_path_corrs = r'C:\Users\Eshel\workspace\MAP\models\corr_predicition'
    file_name_model_corrs = 'model_num_672478_batch_size_128_curr_lr_0.001_num_moms_corrs_5_nn_archi_1_max_lag_2_max_power_1_2_max_power_2_2.pkl'

    net_corrs.load_state_dict(torch.load(os.path.join(model_path_corrs, file_name_model_corrs)))
    net_corrs = net_corrs.to(device)
    ## Get pred moms:

    moms_pred = net_moms(tot_inp.reshape(1, -1).to(device))
    corrs_pred = net_corrs(tot_inp.reshape(1, -1).to(device))

    moms_scaled = 1 / (1 / torch.exp(tot_inp[0]) + 1 / torch.exp(tot_inp[13]))

    moms = [1]

    for mom in range(1, 5):
        moms.append(torch.exp(moms_pred[0, mom - 1]) / (moms_scaled ** (mom + 1)))

    moms_torch = torch.tensor(moms).to(device)
    print('###########################')
    print(100 * torch.abs(torch.tensor(np.exp(res_map[:5])).to(device) - moms_torch) / moms_torch)
    print(torch.abs(torch.tensor(res_map[10:12]).to(device) - corrs_pred.flatten()[:2]))
    print('###########################')

    return moms_pred, corrs_pred, rho, SCV_1, SCV_2, SCV_ser, rho_a, rho_b

if __name__ == '__main__':


    for ind in range(2000):
        if True:

            moms_pred, corrs_pred, rho, SCV_1, SCV_2, SCV_ser, rho_a, rho_b = merge_two_random_stream()

            print('stopping')





            # moms_agg = torch.log(moms_torch)
            #
            # inp_g_gi_1 = torch.concatenate((moms_agg, corrs_pred.flatten())).reshape(1, -1)

            # services_norms2 = services_norms * meanser
            # services_norms.mean()

            # ser_moms_normalized = []
            #
            # for mom in range(1, 6):
            #     ser_moms_normalized.append((services_norms2 ** mom).mean())


            # class Net_steady_1(nn.Module):
            #
            #     def __init__(self, input_size, output_size):
            #         super().__init__()
            #
            #         self.fc1 = nn.Linear(input_size, 50)
            #         self.fc2 = nn.Linear(50, 70)
            #         self.fc3 = nn.Linear(70, 200)
            #         self.fc4 = nn.Linear(200, 350)
            #         self.fc5 = nn.Linear(350, 600)
            #         self.fc6 = nn.Linear(600, output_size)
            #
            #     def forward(self, x):
            #         x = F.relu(self.fc1(x))
            #         x = F.relu(self.fc2(x))
            #         x = F.relu(self.fc3(x))
            #         x = F.relu(self.fc4(x))
            #         x = F.relu(self.fc5(x))
            #         x = self.fc6(x)
            #         return x





            # input_size_steady_1 = 18
            # output_size_steady_1 = 200
            #
            # file_name_model = 'for_tier_1.pkl'
            # model_path = r'C:\Users\Eshel\workspace\data\models\steady_1'
            #
            #
            # model_path_steady_1 = r'C:\Users\Eshel\workspace\Tandem_queueing_ML\models\steady_1'
            # model_path_steady_1 = r'C:\Users\Eshel\workspace\data\models\steady_1'
            #
            # models_steady_1 = os.listdir(model_path_steady_1)
            #
            # full_path_steady_1 = os.path.join(model_path_steady_1, file_name_model) #models_steady_1[0]
            #
            #
            # net_steady_1 = Net_steady_1(input_size_steady_1, output_size_steady_1).to(device)
            # net_steady_1.load_state_dict(torch.load(full_path_steady_1))
            #
            # input_steady_1 = torch.concatenate((inp_g_gi_1, ser_torch), axis=1)
            #
            # m = nn.Softmax(dim=1)
            # # input_steady_1 = inp_g_gi_1
            #
            # probs_steady_1 = net_steady_1(input_steady_1)
            # normalizing_const = torch.exp(input_steady_1[0, -5])
            # probs_steady_1 = m(probs_steady_1)
            # probs_steady_1 = probs_steady_1 * normalizing_const
            #
            # probs_steady_1 = probs_steady_1.to('cpu')
            # probs_steady_1 = torch.concatenate((torch.tensor([[1 - normalizing_const]]), probs_steady_1[0:1, :]), axis=1)

            # with torch.no_grad():
            #     label = probs_steady_1.cpu().numpy().flatten()
            #
            # import numpy as np
            # import matplotlib.pyplot as plt
            #
            # # x values
            # x = np.arange(26)  # 0,1,...,20
            #
            # # take first 21 values
            # y1 = label[:26]
            # y2 = L_dist[:26]
            #
            # bar_width = 0.4
            #
            # # plt.figure()
            # # plt.bar(x - bar_width / 2, y1, width=bar_width, label='Simulation')
            # # plt.bar(x + bar_width / 2, y2, width=bar_width, label='NN')
            # #
            # # plt.xlabel('x')
            # # plt.ylabel('Value')
            # # plt.xticks(x)
            # # plt.legend()
            # # plt.tight_layout()
            # # plt.show()
            # SAE = np.abs(label[:100] - L_dist[:100]).sum()
            # # print(SAE)
            # # print(SCV_1, SCV_2, SCV_ser, rho_a, rho_b, rho)
            # a1 = np.cumsum(arrivals[0])
            # a2 = np.cumsum(arrivals[1])
            #
            # # service = services_norms
            #
            # ES = float(services.mean())
            # VarS = float(services.var(ddof=1))
            #
            # # IDC grid and estimation
            # t_grid = np.linspace(0.5, 80.0, 90)
            # n_windows = 3000
            #
            # rng_idc = np.random.default_rng(123)
            # lam1_hat, idc1 = estimate_idc_curve(a1, t_grid, n_windows, rng_idc)
            # lam2_hat, idc2 = estimate_idc_curve(a2, t_grid, n_windows, rng_idc)
            #
            # lam_hat, idc_sup = idc_superpose(lam1_hat, idc1, lam2_hat, idc2)
            #
            # # Convert IDC_sup(t) to Var(N(t)) using Var(N(t)) = IDC(t) * E[N(t)] = IDC(t) * lam * t
            # varN_sup = idc_sup * (lam_hat * t_grid)
            #
            # # Estimate long-run count variance rate vN from slope of Var(N(t)) vs t
            # vN_hat = estimate_vN_from_var_slope(t_grid, varN_sup, frac_tail=0.4)
            #
            # # RBM mean waiting time / mean number in system
            # rho, vN, vX, EW, EL = rbm_mean_wait(lam_hat, ES, VarS, vN_hat)
            #
            # EL_nn = (np.arange(L_dist.shape[0]) * L_dist).sum()
            # EL_sim = (np.arange(label.shape[0]) * label).sum()
            #
            # error_whitt1 = 100 * abs(EL - EL_sim) / EL_sim
            # error_whitt2 = 100 * abs(EL + rho - (np.arange(L_dist.shape[0]) * L_dist).sum()) / (
            #         np.arange(L_dist.shape[0]) * L_dist).sum()
            #
            # errorwhitt = min(error_whitt1, error_whitt2)
            #
            # error_nn = 100 * abs(EL_nn - EL_sim) / EL_sim
            # print('error_nn {}, errorwhitt {}, SAE {}, rho {}, SCV_1 {} ,SCV_2 {} ,SCV_ser {} , '.format(error_nn, errorwhitt, SAE, rho,SCV_1 ,SCV_2, SCV_ser))
            #
            # if SAE < 0.06:
            #     dump_path = r'C:\Users\Eshel\workspace\MAP\results_queues\tier_1_non_renewal'
            #     pkl.dump((choen_inds, label, L_dist, rho,a, error_nn, errorwhitt,  SCV_1, SCV_2,
            #               SCV_ser, rho_a, rho_b, rho, SAE),
            #              open(os.path.join(dump_path, 'result_{}.pkl'.format(np.random.randint(1,10000))), 'wb'))
            #

        # except:
        #     print('bad iteration',choen_inds, rho)






