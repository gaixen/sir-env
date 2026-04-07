import numpy as np
from scipy.integrate import odeint
import logging
import os

_HUBER_DELTA = 1.0

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_PATH = os.path.dirname(os.path.abspath(__file__))


def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = _HUBER_DELTA) -> float:
    r = np.abs(y_true - y_pred)
    return float(np.sum(np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))))


def loss_sir(S_t, I_t, R_t, S_p, I_p, R_p) -> float:
    return huber(S_t, S_p) + huber(I_t, I_p) + huber(R_t, R_p)


def loss_i(I_t, I_p) -> float:
    return huber(I_t, I_p)


def ode_simple(y, t, beta, gamma):
    Sus, Inf, Rem = y
    dS = -beta * Sus * Inf
    dI = beta * Sus * Inf - gamma * Inf
    dR = gamma * Inf
    return [dS, dI, dR]


def ode_lockdown(y, t, beta, gamma, s_arr):
    Sus, Inf, Rem = y
    idx = min(int(round(t)), len(s_arr) - 1)
    s = s_arr[idx]
    eff = beta * (1.0 - s)
    dS = -eff * Sus * Inf
    dI = eff * Sus * Inf - gamma * Inf
    dR = gamma * Inf
    return [dS, dI, dR]


def ode_lockdown_nu(y, t, beta, gamma, nu_val, s_arr):
    """nu_val can be a scalar (constant) or array (time-varying)."""
    Sus, Inf, Rem = y
    idx = min(int(round(t)), len(s_arr) - 1)
    s = s_arr[idx]
    nu = nu_val[idx] if hasattr(nu_val, "__len__") else nu_val
    eff = beta * (1.0 - s)
    dS = -eff * Sus * Inf - nu * Sus
    dI = eff * Sus * Inf - gamma * Inf
    dR = gamma * Inf + nu * Sus
    return [dS, dI, dR]


def integrate(ode_fn, y0, T, args):
    t_span = np.arange(T, dtype=float)
    return odeint(ode_fn, y0, t_span, args=args, hmax=1.0)
