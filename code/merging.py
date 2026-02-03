import sys
import numpy as np
import os
from tqdm import tqdm
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
from butools.ph import *
from butools.map import *
import torch
import pandas as pd
import pickle as pkl
import os
import numpy as np
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.special import factorial
import numpy as np
import sys
from scipy.linalg import expm, sinm, cosm
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys

sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append(r'C:\Users\Eshel\workspace\one.deep.moment')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/one.deep.moment')

from utils_sample_ph import *
from utils import *
import butools
from butools.ph import *
from butools.fitting import *
from scipy.special import factorial
from numpy.linalg import matrix_power
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.special import factorial
from tqdm import tqdm
import time
import math


def compute_moments(a, T, k, n):
    """ generate first n moments of FT (a, T)
    m_i = ((-1) ** i) i! a T^(-i) 1
    """
    T_in = torch.inverse(T)
    T_powers = torch.eye(k)
    signed_factorial = 1.
    one = torch.ones(k)

    for i in range(1, n + 1):
        signed_factorial *= -i
        T_powers = torch.matmul(T_powers, T_in)  # now T_powers is T^(-i)
        yield signed_factorial * a @ T_powers @ one


def make_ph(lambdas, ps, alpha, k):
    """ Use the arbitrary parameters, and make a valid PT representation  (a, T):
        lambdas: positive size k
        ps: size k x k
        alpha: size k
    """
    ls = lambdas ** 2
    a = torch.nn.functional.softmax(alpha, 0)
    p = torch.nn.functional.softmax(ps, 1)
    lambdas_on_rows = ls.repeat(k, 1).T
    T = (p + torch.diag(-1 - torch.diag(p))) * lambdas_on_rows

    return a, T


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
        moment_list.append(ser_moment_n(s, A, moment).item())
    return np.array(moment_list)


def get_feasible_moments(original_size, n):
    """ Compute feasible k-moments by sampling from high order PH and scaling """
    k = original_size
    ps = torch.randn(k, k)
    lambdas = torch.rand(k) * 100
    alpha = torch.randn(k)
    a, T = make_ph(lambdas, ps, alpha, k)

    # Compute mean
    ms = compute_moments(a, T, k, 1)
    m1 = torch.stack(list(ms)).item()

    # Scale
    T = T * m1
    ms = compute_moments(a, T, k, n)
    momenets = torch.stack(list(ms))

    return a, T, momenets


def get_PH_general_with_zeros(original_size, n1):
    a, T, momenets = get_feasible_moments(original_size, n1)

    matrix = T  # torch.randint(1, 10, (n, n))
    n = T.shape[0]
    # Set the diagonal to non-zero values to avoid diagonal zeros

    # Determine the number of off-diagonal zeros to insert (arbitrary, between 1 and n^2 - n)
    num_zeros = random.randint(1, n ** 2 - n)

    # Get all off-diagonal indices
    off_diagonal_indices = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Randomly select locations for zeros
    zero_indices = random.sample(off_diagonal_indices, num_zeros)

    # Insert zeros at the selected locations
    for i, j in zero_indices:
        matrix[i, j] = 0

    # Print the resulting matrix

    ms = compute_moments(a, T, n, 1)
    m1 = torch.stack(list(ms)).item()

    # Scale
    T = T * m1

    return a, T


def compute_skewness_and_kurtosis_from_raw(m1, m2, m3, m4):
    # Compute central moments
    mu2 = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4

    # Compute skewness and kurtosis
    skewness = mu3 / (mu2 ** 1.5)
    kurtosis = mu4 / (mu2 ** 2)
    excess_kurtosis = kurtosis - 3

    return skewness, kurtosis


def compute_first_lag_auto(D0, D1):
    pis = compute_pi_s(D0, D1)
    ps_steady = ps(D0, D1)

    return (joint_mean(D0, D1) - ph_nth_moment(pis, D0, 1) ** 2) / (
                ph_nth_moment(pis, D0, 2) - ph_nth_moment(pis, D0, 1) ** 2)


def stationary_distribution(Q):
    w, v = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(w))  # eigenvalue closest to 0
    pi = np.real(v[:, idx])
    pi /= pi.sum()
    return pi


def compute_pi_s(D0, D1):
    Q = D0 + D1
    pi = stationary_distribution(Q).reshape(-1, Q.shape[0])

    e = np.ones(D0.shape[0])
    lam = pi @ D1 @ e

    pi_s = (pi @ D1) / lam
    return pi_s


def ps(D0a, D1a):
    return np.linalg.inv(-D0a) @ D1a


def compute_first_lag_auto(D0, D1):
    pis = compute_pi_s(D0, D1)
    ps_steady = ps(D0, D1)

    return (joint_mean(D0, D1) - ph_nth_moment(pis, D0, 1) ** 2) / (
                ph_nth_moment(pis, D0, 2) - ph_nth_moment(pis, D0, 1) ** 2)


def lag1_autocorr_np(x):
    x = np.asarray(x)
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def joint_mean(d0, d1):
    pis = compute_pi_s(d0, d1)
    ps_steady = ps(d0, d1)

    return pis @ np.linalg.inv(-d0) @ ps_steady @ np.linalg.inv(-d0) @ np.ones(d0.shape[0]).reshape(-1, 1)


import numpy as np
import math


def ph_nth_moment(alpha, S, n: int) -> float:
    """
    Compute E[T^n] for a Phase-Type distribution PH(alpha, S).

    Parameters
    ----------
    alpha : array_like, shape (m,) or (1,m)
        Initial probability row vector over transient states.
    S : array_like, shape (m,m)
        Subgenerator matrix (diagonal negative, off-diagonal nonnegative).
    n : int
        Moment order (n >= 1).

    Returns
    -------
    float
        The n-th moment E[T^n].
    """
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer (n >= 1).")

    alpha = np.asarray(alpha, dtype=float).reshape(1, -1)
    S = np.asarray(S, dtype=float)

    m = S.shape[0]
    if S.shape != (m, m):
        raise ValueError("S must be a square (m x m) matrix.")
    if alpha.shape[1] != m:
        raise ValueError("alpha must have length m (same as S dimension).")

    e = np.ones((m, 1))

    # Compute (-S)^(-n) via repeated solves (more stable than explicit inverse)
    A = -S
    X = e.copy()
    for _ in range(n):
        X = np.linalg.solve(A, X)  # X <- A^{-1} X, repeated n times => A^{-n} e

    moment = math.factorial(n) * float(alpha @ X)
    return np.abs(moment)

def anti_identity(n):
    return np.fliplr(np.eye(n))

def make_reversible_Q(pi, eps=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    m = len(pi)

    # symmetric positive rates R_ij
    R = rng.uniform(0.0, 1.0, size=(m, m))
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 0.0)

    Q = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                Q[i, j] = eps * R[i, j] / pi[i]
        Q[i, i] = -Q[i].sum()

    return Q

def sample_mmpp_as_map(m=4, lam_min=0.5, lam_max=20.0, eps=0.05, a=0.3, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    # versatile stationary distribution
    pi = rng.dirichlet(a * np.ones(m))

    # slow-switching CTMC with stationary pi
    Q = make_reversible_Q(pi, eps=eps, rng=rng)

    # high-contrast arrival rates (log-uniform)
    loglam = rng.uniform(np.log(lam_min), np.log(lam_max), size=m)
    lam = np.exp(loglam)

    D1 = np.diag(lam)
    D0 = Q - D1
    return D0, D1, pi, lam

def map_rho1_interarrivals(D0, D1):
    import numpy as np

    def stationary_distribution(Q):
        w, v = np.linalg.eig(Q.T)
        idx = np.argmin(np.abs(w))
        pi = np.real(v[:, idx])
        return pi / pi.sum()

    m = D0.shape[0]
    e = np.ones(m)

    Q = D0 + D1
    pi = stationary_distribution(Q)

    lam = pi @ D1 @ e
    alpha = (pi @ D1) / lam

    A = -D0

    x1 = np.linalg.solve(A, e)
    x2 = np.linalg.solve(A, x1)

    ET = alpha @ x1
    ET2 = 2 * (alpha @ x2)
    VarT = ET2 - ET**2

    y = D1 @ x1
    x_cross = np.linalg.solve(A, y)
    ETnTn1 = (alpha @ x_cross) / lam

    rho1 = (ETnTn1 - ET**2) / VarT
    return float(rho1)
import numpy as np
from numpy.linalg import inv, eig

def _erlang_T(k: int, mean: float) -> np.ndarray:
    """
    Erlang-k PH subgenerator with mean=mean.
    Rate r = k/mean. Bidiagonal: diag -r, superdiag r.
    """
    r = k / mean
    T = np.zeros((k, k), dtype=float)
    for i in range(k):
        T[i, i] = -r
        if i < k - 1:
            T[i, i + 1] = r
    return T

def build_map_two_erlang_regimes(
    k: int = 40,
    mean_fast: float = 0.05,
    mean_slow: float = 5.0,
    eps_fast: float = 1e-6,   # switching prob from fast at arrivals
    eps_slow: float = 1e-6,   # switching prob from slow at arrivals
):
    """
    MAP of size 2k:
      - Fast regime: Erlang(k) with mean_fast
      - Slow regime: Erlang(k) with mean_slow
      - Switching occurs only at arrivals, with small probabilities eps_*
    """
    Tf = _erlang_T(k, mean_fast)
    Ts = _erlang_T(k, mean_slow)

    n = 2 * k
    D0 = np.zeros((n, n), dtype=float)
    D1 = np.zeros((n, n), dtype=float)

    # place Erlang subgenerators into D0
    D0[:k, :k] = Tf
    D0[k:, k:] = Ts

    # rates out of last phases (positive numbers)
    rf = -Tf[-1, -1]
    rs = -Ts[-1, -1]

    # "stay" probs at arrivals
    a = 1.0 - eps_fast  # fast -> fast
    b = 1.0 - eps_slow  # slow -> slow

    # D1 transitions: only from last phase of each Erlang chain
    fast_last = k - 1
    slow_last = 2 * k - 1
    fast_first = 0
    slow_first = k

    # arrival at end of fast chain:
    D1[fast_last, fast_first] = a * rf
    D1[fast_last, slow_first] = (1.0 - a) * rf

    # arrival at end of slow chain:
    D1[slow_last, slow_first] = b * rs
    D1[slow_last, fast_first] = (1.0 - b) * rs

    return D0, D1

def rho1_map(D0: np.ndarray, D1: np.ndarray) -> float:
    """
    Lag-1 correlation of interarrival times using joint-moment formula:
      E[X0 X1] = pi (-D0)^-1 P (-D0)^-1 1,  P = (-D0)^-1 D1
    """
    n = D0.shape[0]
    e = np.ones((n, 1))

    invD0 = inv(-D0)
    P = invD0 @ D1

    # stationary distribution of P
    w, v = eig(P.T)
    vec = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = (vec / vec.sum()).reshape(1, -1)

    mu = (pi @ invD0 @ e)[0, 0]
    EX2 = (2.0 * (pi @ invD0 @ invD0 @ e)[0, 0])
    var = EX2 - mu**2

    EX0X1 = (pi @ invD0 @ P @ invD0 @ e)[0, 0]
    rho1 = (EX0X1 - mu**2) / var
    return float(rho1)


import numpy as np
from numpy.linalg import inv, eig

def _erlang_T(k: int, mean: float) -> np.ndarray:
    """
    Erlang-k PH subgenerator with specified mean.
    Rate r = k/mean. Bidiagonal: diag -r, superdiag r.
    """
    r = k / mean
    T = np.zeros((k, k), dtype=float)
    for i in range(k):
        T[i, i] = -r
        if i < k - 1:
            T[i, i + 1] = r
    return T

def build_map_two_erlang_regimes_negative(
    k: int = 40,
    mean_fast: float = 0.05,
    mean_slow: float = 5.0,
    eps_fast_stay: float = 1e-8,  # fast->fast at arrivals (rare)
    eps_slow_stay: float = 1e-8,  # slow->slow at arrivals (rare)
):
    """
    MAP of size 2k with strong NEGATIVE lag-1 correlation:
      - Fast regime: Erlang(k) with mean_fast
      - Slow regime: Erlang(k) with mean_slow
      - At arrivals, you almost always SWITCH regimes:
          fast_last -> slow_first with prob 1-eps_fast_stay
          slow_last -> fast_first with prob 1-eps_slow_stay
    """
    Tf = _erlang_T(k, mean_fast)
    Ts = _erlang_T(k, mean_slow)

    n = 2 * k
    D0 = np.zeros((n, n), dtype=float)
    D1 = np.zeros((n, n), dtype=float)

    # D0 block diagonal
    D0[:k, :k] = Tf
    D0[k:, k:] = Ts

    # exit rates from last phases
    rf = -Tf[-1, -1]
    rs = -Ts[-1, -1]

    fast_last = k - 1
    slow_last = 2 * k - 1
    fast_first = 0
    slow_first = k

    # At arrival from fast_last: mostly go to slow_first
    D1[fast_last, slow_first] = (1.0 - eps_fast_stay) * rf
    D1[fast_last, fast_first] = eps_fast_stay * rf

    # At arrival from slow_last: mostly go to fast_first
    D1[slow_last, fast_first] = (1.0 - eps_slow_stay) * rs
    D1[slow_last, slow_first] = eps_slow_stay * rs

    return D0, D1

def rho1_map(D0: np.ndarray, D1: np.ndarray) -> float:
    """
    Lag-1 correlation of interarrival times using:
      P = (-D0)^-1 D1
      E[X] = pi (-D0)^-1 1
      E[X^2] = 2 pi (-D0)^-2 1
      E[X0 X1] = pi (-D0)^-1 P (-D0)^-1 1
    """
    n = D0.shape[0]
    e = np.ones((n, 1))

    invD0 = inv(-D0)
    P = invD0 @ D1

    # stationary distribution of P
    w, v = eig(P.T)
    vec = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = (vec / vec.sum()).reshape(1, -1)

    mu = (pi @ invD0 @ e)[0, 0]
    EX2 = (2.0 * (pi @ invD0 @ invD0 @ e)[0, 0])
    var = EX2 - mu**2

    EX0X1 = (pi @ invD0 @ P @ invD0 @ e)[0, 0]
    rho1 = (EX0X1 - mu**2) / var
    return float(rho1)




def stationary_distribution(Q):
    w, v = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(w))  # eigenvalue closest to 0
    pi = np.real(v[:, idx])
    pi /= pi.sum()
    return pi

def map_rho1_interarrivals(D0, D1):
    """
    Lag-1 autocorrelation of interarrival times T_n for MAP(D0, D1).
    """
    m = D0.shape[0]
    e = np.ones(m)

    Q = D0 + D1
    pi = stationary_distribution(Q)

    lam = pi @ D1 @ e
    if lam <= 0:
        raise ValueError("Non-positive arrival rate; check D0, D1.")

    alpha = (pi @ D1) / lam  # row vector length m

    # Use solves for stability instead of explicit inverse
    A = -D0  # should be nonsingular for a valid MAP

    # (-D0)^(-1) e
    x1 = np.linalg.solve(A, e)
    # (-D0)^(-2) e
    x2 = np.linalg.solve(A, x1)

    ET  = alpha @ x1
    ET2 = 2.0 * (alpha @ x2)
    VarT = ET2 - ET**2
    if VarT <= 0:
        raise ValueError("Non-positive variance; MAP may be degenerate or ill-conditioned.")

    # cross moment E[T_n T_{n+1}]
    y = D1 @ x1                  # D1 (-D0)^(-1) e
    x_cross = np.linalg.solve(A, y)  # (-D0)^(-1) D1 (-D0)^(-1) e
    ETnTn1 = (alpha @ x_cross) / lam

    rho1 = (ETnTn1 - ET**2) / VarT
    return float(rho1)


def sample_coxian(degree, max_rate):
    # print(degree)
    lambdas_ = np.random.rand(degree) * max_rate
    ps_ = np.random.rand(degree - 1)
    A = np.diag(-lambdas_) + np.diag(lambdas_[:degree-1] * ps_, k=1)
    alpha = np.eye(degree)[[0]]
    mean_val  = ser_moment_n(alpha, A, 1)
    A1 = A*mean_val.item()
    lambdas_ = lambdas_*mean_val.item()
    return alpha, A1, lambdas_, ps_

def stationary_distribution(Q):
    w, v = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(w))
    pi = np.real(v[:, idx])
    pi /= pi.sum()
    return pi

def map_lag1_autocorrelation(D0, D1):
    m = D0.shape[0]
    e = np.ones(m)

    Q = D0 + D1

    # stationary distribution of background chain
    pi = stationary_distribution(Q).reshape(-1)

    # arrival rate
    lam = pi @ D1 @ e

    # initial PH vector
    alpha = (pi @ D1) / lam

    # matrix inverses
    D0_inv = np.linalg.inv(-D0)
    D0_inv2 = D0_inv @ D0_inv

    # moments
    ET = alpha @ D0_inv @ e
    ET2 = 2 * alpha @ D0_inv2 @ e
    VarT = ET2 - ET**2

    # cross moment
    ETnTn1 = (alpha @ D0_inv @ D1 @ D0_inv @ e) / lam

    rho1 = (ETnTn1 - ET**2) / VarT

    return float(rho1)

def ph_to_map_renewal(alpha, T):
    """
    Convert PH(alpha,T) to a MAP(D0,D1) that generates a renewal process
    with i.i.d. PH inter-arrival times.
    alpha: shape (m,) or (1,m)
    T:     shape (m,m)
    """
    alpha = np.asarray(alpha, dtype=float).reshape(1, -1)
    T = np.asarray(T, dtype=float)
    m = T.shape[0]
    assert T.shape == (m, m)

    ones = np.ones((m, 1))
    t = -T @ ones                 # column vector exit rates
    D0 = T
    D1 = t @ alpha                # rank-1

    return D0, D1


def ph_to_map_renewal(alpha, T, *, check=True, tol=1e-10):
    """
    Convert a PH distribution PH(alpha, T) into a MAP (D0, D1)
    that generates a renewal process with i.i.d. PH inter-arrival times.

    Construction:
        t  = -T * 1
        D0 = T
        D1 = t * alpha

    Parameters
    ----------
    alpha : array_like
        Initial probability row vector (shape (m,) or (1,m)).
    T : array_like
        PH subgenerator matrix (shape (m,m)).
    check : bool
        If True, run basic validity checks.
    tol : float
        Tolerance used in checks.

    Returns
    -------
    D0, D1 : np.ndarray
        MAP representation matrices of shape (m,m).
    """
    alpha = np.asarray(alpha, dtype=float).reshape(1, -1)
    T = np.asarray(T, dtype=float)

    m = T.shape[0]
    if T.shape != (m, m):
        raise ValueError("T must be square (m,m).")
    if alpha.shape != (1, m):
        raise ValueError(f"alpha must have length m={m}.")

    ones = np.ones((m, 1))
    t = -T @ ones                  # exit (absorption) rates column vector

    D0 = T.copy()
    D1 = t @ alpha                 # rank-1 matrix

    if check:
        # alpha should be a probability vector
        if np.any(alpha < -tol):
            raise ValueError("alpha has negative entries.")
        s = float(alpha.sum())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"alpha should sum to 1 (got {s}).")

        # T should have nonnegative off-diagonals and negative diagonals
        off = T - np.diag(np.diag(T))
        if np.any(off < -tol):
            raise ValueError("T has negative off-diagonal entries.")
        if np.any(np.diag(T) >= tol):
            raise ValueError("T diagonal entries should be negative.")

        # MAP generator Q = D0 + D1 must have row sums 0
        Q = D0 + D1
        row_sums = Q @ np.ones((m, 1))
        if np.max(np.abs(row_sums)) > 1e-7:
            raise ValueError("Invalid MAP: rows of (D0+D1) do not sum to 0.")

        # D1 should be nonnegative
        if np.any(D1 < -tol):
            raise ValueError("D1 has negative entries.")

    return D0, D1


def map_nth_moment(D0, D1, n):
    """
    Compute E[X^n] for a MAP(D0,D1).
    """
    m = D0.shape[0]
    e = np.ones((m, 1))

    invD0 = np.linalg.inv(-D0)
    P = invD0 @ D1

    # stationary distribution of P
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    pi = pi.reshape(1, -1)

    moment = math.factorial(n) * (pi @ np.linalg.matrix_power(invD0, n) @ e)[0, 0]
    return moment



def stationary_dist_dtmc(P):
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    return pi / pi.sum()

def joint_moment_map(D0, D1, i, ell, j):
    """
    E[X_0^i X_ell^j] for a MAP(D0,D1) in stationarity.
    """
    m = D0.shape[0]
    e = np.ones((m, 1))

    invD0 = inv(-D0)
    P = invD0 @ D1
    pi = stationary_dist_dtmc(P).reshape(1, -1)

    A_i = np.linalg.matrix_power(invD0, i)   # (-D0)^-i
    A_j = np.linalg.matrix_power(invD0, j)   # (-D0)^-j
    P_ell = np.linalg.matrix_power(P, ell)

    return (math.factorial(i) * math.factorial(j) *
            (pi @ A_i @ P_ell @ A_j @ e)[0, 0])



def stationary_dist_dtmc(P):
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    return pi.reshape(1, -1)

def map_power_corr(D0, D1, k=1, i=1, j=1):
    """
    Corr(X_0^i, X_k^j) for stationary inter-arrival sequence of a MAP(D0,D1).
    """
    m = D0.shape[0]
    e = np.ones((m, 1))

    invD0 = np.linalg.inv(-D0)
    P = invD0 @ D1
    pi = stationary_dist_dtmc(P)

    # helper: E[X^r] = r! * pi * (-D0)^(-r) * 1
    def EX_pow(r):
        A = np.linalg.matrix_power(invD0, r)
        return float(math.factorial(r) * (pi @ A @ e))

    # joint: E[X0^i Xk^j] = i! j! pi (-D0)^(-i) P^k (-D0)^(-j) 1
    A_i = np.linalg.matrix_power(invD0, i)
    A_j = np.linalg.matrix_power(invD0, j)
    Pk  = np.linalg.matrix_power(P, k)

    EX0i_Xkj = float(math.factorial(i) * math.factorial(j) *
                     (pi @ A_i @ Pk @ A_j @ e))

    EXi  = EX_pow(i)
    EXj  = EX_pow(j)
    EX2i = EX_pow(2*i)
    EX2j = EX_pow(2*j)

    var_i = EX2i - EXi**2
    var_j = EX2j - EXj**2
    if var_i <= 0 or var_j <= 0:
        return 0.0

    corr = (EX0i_Xkj - EXi * EXj) / np.sqrt(var_i * var_j)
    return float(corr)


def stationary_dist_ctmc(Q, tol=1e-12):
    """
    Stationary distribution pi of an ergodic CTMC generator Q (rows sum to 0).
    Solves pi Q = 0, pi 1 = 1 via least-squares.
    """
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= tol:
        # fallback: uniform
        return np.ones(n) / n
    return pi / s

def stationary_dist_dtmc(P, tol=1e-12):
    """
    Stationary distribution of a stochastic matrix P (rows sum to 1).
    """
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= tol:
        return np.ones(n) / n
    return pi / s

# ----------------------------
# MAP moments + rho1
# ----------------------------
def map_moments_and_rho1(D0, D1):
    """
    Returns mean, scv, skewness, excess_kurtosis, rho1 for inter-arrival times.
    Uses standard MAP joint-moment formulas:
      P = (-D0)^-1 D1
      E[X^k] = k! * pi * (-D0)^-k * 1
      E[X0 X1] = pi * (-D0)^-1 * P * (-D0)^-1 * 1
    where pi is stationary of P (arrival-epoch chain).
    """
    n = D0.shape[0]
    e = np.ones((n, 1))
    invD0 = np.linalg.inv(-D0)
    P = invD0 @ D1

    pi = stationary_dist_dtmc(P).reshape(1, -1)

    # raw moments
    EX1 = (pi @ invD0 @ e)[0, 0]
    EX2 = (2.0 * (pi @ invD0 @ invD0 @ e)[0, 0])
    EX3 = (6.0 * (pi @ invD0 @ invD0 @ invD0 @ e)[0, 0])
    EX4 = (24.0 * (pi @ invD0 @ invD0 @ invD0 @ invD0 @ e)[0, 0])

    var = EX2 - EX1**2
    scv = var / (EX1**2)

    # central moments for skew/kurt
    mu3 = EX3 - 3*EX2*EX1 + 2*EX1**3
    mu4 = EX4 - 4*EX3*EX1 + 6*EX2*EX1**2 - 3*EX1**4
    skew = mu3 / (var**1.5) if var > 0 else 0.0
    excess_kurt = mu4 / (var**2) - 3.0 if var > 0 else 0.0

    # lag-1 correlation
    EX0X1 = (pi @ invD0 @ P @ invD0 @ e)[0, 0]
    rho1 = (EX0X1 - EX1**2) / var if var > 0 else 0.0

    return float(EX1), float(scv), float(skew), float(excess_kurt), float(rho1)

# ----------------------------
# Random MAP(n) generator
# ----------------------------
def random_map(
    n: int,
    rng=None,
    corr_strength: float = 0.2,     # desired sign+strength heuristic in [-1,1]
    corr_cap: float = 0.25,         # try to keep |rho1| <= corr_cap
    heavy_tail_mix: float = 0.4,    # more -> more diverse rates -> more diverse SCV/skew/kurt
    max_tries: int = 2000,
):
    """
    Generates a random valid MAP (D0, D1) of size n with mean inter-arrival = 1.
    Also tries (heuristically) to keep lag-1 autocorrelation within corr_cap.

    corr_strength > 0 tends to produce positive rho1
    corr_strength < 0 tends to produce negative rho1
    corr_strength ~= 0 tends to produce near-renewal

    heavy_tail_mix controls how spread the exit rates are (versatility of moments).
    """
    if rng is None:
        rng = np.random.default_rng()

    corr_strength = float(np.clip(corr_strength, -1.0, 1.0))
    corr_cap = float(abs(corr_cap))

    # permutation for "alternating" tendency (negative correlation)
    perm = np.arange(n)
    rng.shuffle(perm)
    # ensure no fixed points as much as possible
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    I = np.eye(n)
    Alt = np.zeros((n, n))
    Alt[np.arange(n), perm] = 1.0

    for _ in range(max_tries):
        # ---- Step A: choose total exit rates v_i (spread controls tail/moments)
        # log-uniform-ish mixture to allow very slow and very fast states
        # heavy_tail_mix increases spread by mixing two scales
        base = rng.lognormal(mean=0.0, sigma=1.0, size=n)
        spread = rng.lognormal(mean=0.0, sigma=2.0, size=n)
        v = (1.0 - heavy_tail_mix) * base + heavy_tail_mix * spread
        v = np.clip(v, 1e-3, 1e3)

        # ---- Step B: choose arrival probability per departure (p_i in (0,1))
        # This determines arrival rates a_i = p_i v_i
        p_arr = rng.beta(a=1.2, b=1.2, size=n)
        p_arr = np.clip(p_arr, 1e-4, 1.0 - 1e-4)

        a = p_arr * v                # arrival rates from each state
        na = (1.0 - p_arr) * v       # non-arrival transition rates from each state

        # ---- Step C: choose non-arrival routing R (no self loops for D0 off-diag)
        R = rng.random((n, n))
        np.fill_diagonal(R, 0.0)
        R = R / R.sum(axis=1, keepdims=True)

        # ---- Step D: choose arrival routing S to induce correlation
        # Base random routing
        U = rng.random((n, n))
        U = U / U.sum(axis=1, keepdims=True)

        # Mix for correlation:
        #  - positive: bias toward staying in same state on arrival (I)
        #  - negative: bias toward jumping via permutation Alt
        #  - plus some random U to keep versatility/ergodicity
        eta = min(0.98, abs(corr_strength))  # correlation bias weight
        if corr_strength >= 0:
            S = (1 - eta) * U + eta * I
        else:
            S = (1 - eta) * U + eta * Alt
        S = np.clip(S, 0.0, None)
        S = S / S.sum(axis=1, keepdims=True)

        # ---- Build D0 and D1
        D0 = np.zeros((n, n))
        D1 = np.zeros((n, n))

        # Off-diagonals
        D0 += (na[:, None] * R)      # non-arrival jumps
        D1 += (a[:, None] * S)       # arrival jumps

        # Diagonals: total out rate is v_i (includes both arrival and non-arrival)
        np.fill_diagonal(D0, -v)

        # Validity check: rows of Q=D0+D1 sum to ~0
        Q = D0 + D1
        if np.max(np.abs(Q.sum(axis=1))) > 1e-8:
            continue

        # ---- Scale to mean 1
        # Arrival rate lambda = pi_Q * D1 * 1, where pi_Q is stationary of CTMC Q
        piQ = stationary_dist_ctmc(Q)
        lam = float(piQ @ D1 @ np.ones(n))
        if lam <= 1e-12:
            continue

        mean_current = 1.0 / lam
        scale = mean_current  # multiplying rates by scale makes mean -> mean/scale = 1
        D0s = D0 * scale
        D1s = D1 * scale

        return (D0s, D1s)

        # ---- Evaluate moments + rho1
        mean, scv, skew, exkurt, rho1 = map_moments_and_rho1(D0s, D1s)

        # hard enforce mean=1 numerically
        if not (0.999 <= mean <= 1.001):
            continue

        # If you want rho to be mild (<= corr_cap), accept only if it meets bound.
        # If you *don't* care, set corr_cap large.
        if abs(rho1) <= corr_cap:
            return D0s, D1s, dict(mean=mean, scv=scv, skew=skew, excess_kurtosis=exkurt, rho1=rho1)

    raise RuntimeError("Failed to sample a MAP meeting the constraints; try increasing max_tries or loosening corr_cap.")


import numpy as np
from numpy.linalg import inv, eig


# ----------------------------
# Stationary distributions
# ----------------------------
def stationary_dist_ctmc(Q, tol=1e-12):
    """
    Stationary distribution pi of an ergodic CTMC generator Q (rows sum to 0).
    Solves pi Q = 0, pi 1 = 1 via least-squares.
    """
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= tol:
        return np.ones(n) / n
    return pi / s


def stationary_dist_dtmc(P, tol=1e-12):
    """
    Stationary distribution of a stochastic matrix P (rows sum to 1).
    """
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= tol:
        return np.ones(n) / n
    return pi / s


# ----------------------------
# MAP moments + rho1
# ----------------------------
def map_moments_and_rho1(D0, D1):
    """
    Returns mean, scv, skewness, excess_kurtosis, rho1 for inter-arrival times.

    Uses:
      P = (-D0)^-1 D1
      E[X^k] = k! * pi * (-D0)^-k * 1
      E[X0 X1] = pi * (-D0)^-1 * P * (-D0)^-1 * 1
    where pi is stationary of P (arrival-epoch chain).
    """
    n = D0.shape[0]
    e = np.ones((n, 1))
    invD0 = inv(-D0)
    P = invD0 @ D1

    pi = stationary_dist_dtmc(P).reshape(1, -1)

    EX1 = (pi @ invD0 @ e)[0, 0]
    EX2 = (2.0 * (pi @ invD0 @ invD0 @ e)[0, 0])
    EX3 = (6.0 * (pi @ invD0 @ invD0 @ invD0 @ e)[0, 0])
    EX4 = (24.0 * (pi @ invD0 @ invD0 @ invD0 @ invD0 @ e)[0, 0])

    var = EX2 - EX1 ** 2
    scv = var / (EX1 ** 2)

    mu3 = EX3 - 3 * EX2 * EX1 + 2 * EX1 ** 3
    mu4 = EX4 - 4 * EX3 * EX1 + 6 * EX2 * EX1 ** 2 - 3 * EX1 ** 4
    skew = mu3 / (var ** 1.5) if var > 1e-15 else 0.0
    excess_kurt = mu4 / (var ** 2) - 3.0 if var > 1e-15 else 0.0

    EX0X1 = (pi @ invD0 @ P @ invD0 @ e)[0, 0]
    rho1 = (EX0X1 - EX1 ** 2) / var if var > 1e-15 else 0.0

    return float(EX1), float(EX2), float(EX3), float(EX4), float(rho1)


# ----------------------------
# Random MAP(n) generator (NEGATIVE rho1)
# ----------------------------
def random_map_negative(
        n: int,
        rng=None,
        neg_strength: float = 0.6,  # 0..1 how strongly we "alternate" after arrivals
        rho_cap: float = 0.25,  # enforce |rho1| <= rho_cap (set bigger to allow more)
        want_rho_at_most: float = -0.02,  # accept only if rho1 <= this (negative)
        heavy_tail_mix: float = 0.6,  # more -> more diverse SCV/skew/kurt
        max_tries: int = 8000,
):
    """
    Generates a random MAP(n) with:
      - mean inter-arrival = 1 (by scaling)
      - flexible SCV/skew/kurt (via heavy_tail_mix)
      - mild NEGATIVE lag-1 correlation (rho1 <= want_rho_at_most and |rho1| <= rho_cap)

    neg_strength controls how much arrival transitions are biased to "alternate"
    to a different state (permutation mapping).
    """
    if rng is None:
        rng = np.random.default_rng()

    neg_strength = float(np.clip(neg_strength, 0.0, 0.999))
    rho_cap = float(abs(rho_cap))
    want_rho_at_most = float(want_rho_at_most)

    # Build a permutation with (almost) no fixed points for alternation
    perm = np.arange(n)
    rng.shuffle(perm)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    Alt = np.zeros((n, n))
    Alt[np.arange(n), perm] = 1.0

    I = np.eye(n)

    for _ in range(max_tries):
        # ---- A) total exit rates v_i (spread -> distribution versatility)
        base = rng.lognormal(mean=0.0, sigma=1.0, size=n)
        spread = rng.lognormal(mean=0.0, sigma=2.2, size=n)
        v = (1.0 - heavy_tail_mix) * base + heavy_tail_mix * spread
        v = np.clip(v, 1e-3, 1e3)

        # ---- B) fraction of departures that are arrivals, p_arr in (0,1)
        p_arr = rng.beta(a=1.2, b=1.2, size=n)
        p_arr = np.clip(p_arr, 1e-4, 1.0 - 1e-4)

        a = p_arr * v
        na = (1.0 - p_arr) * v

        # ---- C) non-arrival routing R (random)
        R = rng.random((n, n))
        np.fill_diagonal(R, 0.0)
        R = R / R.sum(axis=1, keepdims=True)

        # ---- D) arrival routing S with NEGATIVE-corr bias
        # Mix: mostly Alt (forces alternation) + some random routing to keep ergodicity/variety
        U = rng.random((n, n))
        U = U / U.sum(axis=1, keepdims=True)

        S = (1.0 - neg_strength) * U + neg_strength * Alt
        S = np.clip(S, 0.0, None)
        S = S / S.sum(axis=1, keepdims=True)

        # ---- Build D0/D1
        D0 = (na[:, None] * R)
        D1 = (a[:, None] * S)
        np.fill_diagonal(D0, -v)

        Q = D0 + D1
        if np.max(np.abs(Q.sum(axis=1))) > 1e-8:
            continue

        # ---- Scale to mean 1 using CTMC stationary of Q: lambda = piQ D1 1, mean=1/lambda
        piQ = stationary_dist_ctmc(Q)
        lam = float(piQ @ D1 @ np.ones(n))
        if lam <= 1e-12 or not np.isfinite(lam):
            continue

        scale = 1.0 / lam  # multiply all rates by scale => lambda becomes 1 => mean becomes 1
        D0s = D0 * scale
        D1s = D1 * scale

        # ---- Evaluate
        mean, scv, skew, exkurt, rho1 = map_moments_and_rho1(D0s, D1s)

        # numerical mean check
        if not (0.999 <= mean <= 1.001):
            continue

        # enforce: negative but mild
        if (rho1 <= want_rho_at_most) and (abs(rho1) <= rho_cap):
            return (D0s, D1s)  # , dict(mean=mean, scv=scv, skew=skew, excess_kurtosis=exkurt, rho1=rho1)

    raise RuntimeError(
        "Failed to sample a MAP meeting constraints. "
        "Try: increase max_tries, increase rho_cap, or loosen want_rho_at_most toward 0."
    )


def generate_renewal_MAP(max_degree):
    try:
        degree = np.random.randint(10, max_degree)
        option = 1 # np.random.randint(1, 4)
        print(option)
        if option == 1:

            n = np.random.randint(7, max(11, degree))
            a, T = get_PH_general_with_zeros(degree, n)
            a = np.asarray(a, dtype=float).reshape(1, -1)
            a = a / a.sum()
            T = np.array(T)
        elif option == 2:

            a, T, _, _ = sample_coxian(degree=degree, max_rate=20)
            a = np.array(a).flatten()
            T = np.array(T)
        else:

            dat = sample(degree)
            a = dat[0]
            T = dat[1]

        D0, D1 = ph_to_map_renewal(a, T)
        moms2 = map_nth_moment(D0, D1, 2)
        if moms2 < 10:
            generate_renewal_MAP(max_degree)
        else:
            return D0, D1

    except:
        return print('numerical error') #generate_renewal_MAP(max_degree)


def create_mom_cor_vector(D0, D1):
    mom_cors = []

    for mom in range(1, 11):
        mom_cors.append(map_nth_moment(D0, D1, mom))

    for k in range(1, 6):
        for i in range(1, 6):
            for j in range(1, 6):
                mom_cors.append(map_power_corr(D0, D1, k=k, i=i, j=j))
    return np.array(mom_cors)



import numpy as np

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
        if not np.allclose(Qs.sum(axis=1), 0.0, atol=1e-9):
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


## Strong Correlations

def give_strong_pos_cor(degree):
    k = np.random.randint(1, degree)
    mean_fast = np.random.uniform(0.01, 10)
    mean_slow = np.random.uniform(8, 80)
    eps_fast_stay = (10 ** np.random.randint(1, 8)) * 1e-9
    eps_slow_stay = (10 ** np.random.randint(1, 8)) * 1e-9
    # Example: with k ~ 40 and strong separation, rho1 is typically very close to -1
    D0, D1 = build_map_two_erlang_regimes(
        k=k, mean_fast=mean_fast, mean_slow=mean_slow, eps_fast=eps_fast_stay, eps_slow=eps_slow_stay
    )
    mean = map_nth_moment(D0, D1, 1)
    D0 = D0 * mean
    D1 = D1 * mean
    return D0, D1


def give_strong_neg_cor(degree):
    k = np.random.randint(1, degree)
    mean_fast = np.random.uniform(0.01, 10)
    mean_slow = np.random.uniform(8, 80)
    eps_fast_stay = (10 ** np.random.randint(1, 8)) * 1e-9
    eps_slow_stay = (10 ** np.random.randint(1, 8)) * 1e-9
    # Example: with k ~ 40 and strong separation, rho1 is typically very close to -1
    D0, D1 = build_map_two_erlang_regimes_negative(
        k=k, mean_fast=mean_fast, mean_slow=mean_slow, eps_fast_stay=eps_fast_stay, eps_slow_stay=eps_slow_stay
    )

    mean = map_nth_moment(D0.copy(), D1.copy(), 1)

    D0 = D0 * mean
    D1 = D1 * mean

    return D0, D1


def create_single_data_point(low_max_size=15, large_max_size=80):
    option = np.random.randint(1, 6)
    # print('A', option)
    if np.random.rand() < 0.7:
        option == 1
    if option == 1:
        D0a, D1a = generate_renewal_MAP(low_max_size)
    elif option == 2:
        D0a, D1a = give_strong_pos_cor(low_max_size)
    elif option == 3:
        D0a, D1a = give_strong_neg_cor(low_max_size)
    elif option == 4:
        D0a, D1a = random_map(
            n=low_max_size,
            corr_strength=0.75,  # try positive mild correlation
            corr_cap=0.5,  # keep |rho1| <= 0.25
            heavy_tail_mix=0.6,  # more variability in SCV/skew/kurt
            max_tries=5000,
        )
    elif option == 5:
        D0a, D1a = random_map_negative(
            n=low_max_size,
            neg_strength=0.85,  # stronger alternation tendency
            rho_cap=0.99,  # keep |rho1| <= 0.25
            want_rho_at_most=-0.031,  # ensure negative
            heavy_tail_mix=0.7,
            max_tries=15000
        )
    scale_low = 1 / np.random.uniform(0.01, 1)
    D0a = D0a * scale_low
    D1a = D1a * scale_low
    # print(map_nth_moment(D0a, D1a, 1),scale_low)
    # print('B', option,)
    option = np.random.randint(1, 6)
    if np.random.rand() < 0.7:
        option == 1
    if option == 1:
        D0b, D1b = generate_renewal_MAP(large_max_size)
    elif option == 2:
        D0b, D1b = give_strong_pos_cor(large_max_size)
    elif option == 3:
        D0b, D1b = give_strong_neg_cor(large_max_size)
    elif option == 4:
        D0b, D1b = random_map(
            n=large_max_size,
            corr_strength=0.75,  # try positive mild correlation
            corr_cap=0.5,  # keep |rho1| <= 0.25
            heavy_tail_mix=0.6,  # more variability in SCV/skew/kurt
            max_tries=5000,
        )


    elif option == 5:
        D0b, D1b = random_map_negative(
            n=large_max_size,
            neg_strength=0.85,  # stronger alternation tendency
            rho_cap=0.99,  # keep |rho1| <= 0.25
            want_rho_at_most=-0.031,  # ensure negative
            heavy_tail_mix=0.7,
            max_tries=15000
        )
    # print(map_nth_moment(D0b, D1b, 1))
    # print('compute input')
    # print(D0a.shape, D1a.shape)
    resa = create_mom_cor_vector(D0a.copy(), D1a.copy())
    resb = create_mom_cor_vector(D0b.copy(), D1b.copy())
    # print('merge')

    D0_merged, D1_merged = superpose_map(D0a.copy(), D1a.copy(), D0b.copy(), D1b.copy())
    # print(D0_merged.shape)
    # print('compute output')
    res_merged = create_mom_cor_vector(D0_merged.copy(), D1_merged.copy())
    inp = np.concatenate((resa, resb))
    return (inp, res_merged, D0_merged.shape[0])

if sys.platform == 'linux':
    data_path = '/scratch/eliransc/merge_data'
else:
    data_path = r'C:\Users\Eshel\workspace\data\merge_data'


# for ind in range(1500):
#     try:
#         degree = 100  # np.random.randint(10, max_degree)
#         option = np.random.randint(1, 4)
#         if option == 1:
#
#             n = np.random.randint(10, degree)
#             a, T = get_PH_general_with_zeros(degree, n)
#             a = a / a.sum()
#             a = np.array(a).flatten()
#             T = np.array(T)
#         elif option == 2:
#
#             a, T, _, _ = sample_coxian(degree=degree, max_rate=20)
#             a = np.array(a).flatten()
#             T = np.array(T)
#         else:
#
#             dat = sample(degree)
#             a = dat[0]
#             T = dat[1]
#
#         D0, D1 = ph_to_map_renewal(a, T)
#         mom2 =  map_nth_moment(D0, D1, 2)
#         if mom2 > 10:
#             print(mom2, option)
#     except:
#         pass

for ind in range(5000):
    try:
        now = time.time()
        inp, res_merged, shape = create_single_data_point()
        file_name = 'marging_'+str(ind) + '_sizemerged_' + str(shape)+'_seed_'+str(np.random.randint(1,10000)) + '.pkl'
        full_path = os.path.join(data_path, file_name)
        end = time.time()
        print('Took {} seconds'. format(end -now))
        print(inp[0], inp[135], res_merged[0])
        pkl.dump(( inp, res_merged), open(full_path, 'wb'))
    except:
        print('bad run')