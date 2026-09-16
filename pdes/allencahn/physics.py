from typing import Callable
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# 1D Allen-Cahn: u_t - kappa*u_xx - 5u(1-u^2) = 0, periodic BC on [min_x, max_x], IC u(x,0) = cos(pi*x).
# The Padé approximant's symbolic recursion only encodes the linear diffusion term (kappa*u_xx); the
# nonlinear reaction term 5u(1-u^2) is left entirely to the NN correction (same choice as Burgers).

def symbolic_allencahn() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    x_sym = sp.Symbol("x", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    u_initial = sp.cos(sp.pi * x_sym)
    pde_op = lambda u: kappa_sym * u.diff(x_sym, 2)
    return x_sym, t_sym, kappa_sym, u_initial, pde_op

# Numeric IC matching symbolic_allencahn's u_initial exactly, so the Padé approximant and the reference
# solver's initial condition can never drift out of sync.
def uv_initial_condition(x: np.ndarray) -> np.ndarray:
    return np.cos(np.pi * x)

# BDF method-of-lines reference for u_t - kappa*u_xx - 5u(1-u^2) = 0 with periodic BCs
def solve_ac_ref(x_plot: np.ndarray, t_plot: np.ndarray, kappa_test: float, u0: np.ndarray) -> np.ndarray:
    dx = x_plot[1] - x_plot[0]
    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        lap = (np.roll(u, 1) - 2.0 * u + np.roll(u, -1)) / dx ** 2
        return kappa_test * lap + 5.0 * u * (1.0 - u ** 2)
    sol = solve_ivp(rhs, (0.0, t_plot[-1]), u0, t_eval=t_plot, method="BDF", rtol=1e-8, atol=1e-10)
    return sol.y.T
