from typing import Callable
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# Nonlinear Schrodinger equation: i*w_t + kappa*w_xx + |w|^2*w = 0 for w = u + iv split into real/imaginary components:
#   u_t + kappa*v_xx + (u^2+v^2)*v = 0
#   v_t - kappa*u_xx - (u^2+v^2)*u = 0
DOMAIN_LENGTH = 15.0
IC_MODE = 1
IC_AMPLITUDE = 0.8
IC_MULTIPLIER = 4.0

def uv_initial_condition_mode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = IC_MULTIPLIER * np.pi * IC_MODE / DOMAIN_LENGTH
    return IC_AMPLITUDE * np.cos(k * x), np.zeros_like(x)

def symbolic_linearized_schrodinger() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, sp.Expr, Callable, Callable]:
    x_sym = sp.Symbol("x", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    # Same mode as uv_initial_condition_mode -- both derive from the constants above, since a Pade
    # built for a different IC than the reference solver is seeded with fails silently.
    k_sym = IC_MULTIPLIER * sp.pi * IC_MODE / DOMAIN_LENGTH
    u_initial, v_initial = IC_AMPLITUDE * sp.cos(k_sym * x_sym), sp.Integer(0)
    pde_op_u = lambda v: -kappa_sym * v.diff(x_sym, 2)
    pde_op_v = lambda u: kappa_sym * u.diff(x_sym, 2)
    return x_sym, t_sym, kappa_sym, u_initial, v_initial, pde_op_u, pde_op_v

# RK45 reference solution: central finite difference with periodic BCs for the coupled (u, v) system
def solve_schrodinger_ref(x_plot: np.ndarray, t_plot: np.ndarray, kappa_test: float, u0: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx_eval = len(x_plot)
    dx = x_plot[1] - x_plot[0]
    D2 = np.zeros((nx_eval, nx_eval))
    for i in range(nx_eval):
        D2[i, i] = -2.0
        D2[i, (i - 1) % nx_eval] = 1.0
        D2[i, (i + 1) % nx_eval] = 1.0
    D2 /= dx ** 2

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u, v = y[:nx_eval], y[nx_eval:]
        u_xx, v_xx = D2 @ u, D2 @ v
        h2 = u ** 2 + v ** 2
        return np.concatenate([-kappa_test * v_xx - h2 * v, kappa_test * u_xx + h2 * u])

    sol = solve_ivp(rhs, (0.0, t_plot[-1]), np.concatenate([u0, v0]), t_eval=t_plot, method="RK45", rtol=1e-6, atol=1e-9)
    return sol.y[:nx_eval, :].T, sol.y[nx_eval:, :].T
