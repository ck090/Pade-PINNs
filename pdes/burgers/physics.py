from typing import Callable
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# Burgers' equation: u_t + u*u_x - nu*u_xx = 0, periodic BC on [min_x, max_x], classic shock IC
# u(x,0) = -sin(pi*x) (single compression at x=0, inviscid shock forms at t=1/pi).

def symbolic_burgers() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    x_sym = sp.Symbol("x", real=True)
    t_sym, nu_sym = sp.symbols("t nu", real=True, positive=True)
    u_initial = -sp.sin(sp.pi * x_sym)
    pde_op = lambda u: nu_sym * u.diff(x_sym, 2)
    return x_sym, t_sym, nu_sym, u_initial, pde_op

# Numeric IC matching symbolic_burgers's u_initial exactly, so the Padé approximant and the reference
# solver's initial condition can never drift out of sync.
def uv_initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)

# BDF method-of-lines reference for u_t + u*u_x - nu*u_xx = 0 with periodic BCs
def solve_burgers_ref(x_plot: np.ndarray, t_plot: np.ndarray, nu_test: float, u0: np.ndarray) -> np.ndarray:
    nx_eval = len(x_plot)
    dx = x_plot[1] - x_plot[0]
    D2 = np.zeros((nx_eval, nx_eval)); D1 = np.zeros((nx_eval, nx_eval))
    for i in range(nx_eval):
        D2[i, i] = -2.0; D2[i, (i - 1) % nx_eval] = 1.0; D2[i, (i + 1) % nx_eval] = 1.0
        D1[i, (i + 1) % nx_eval] = 1.0; D1[i, (i - 1) % nx_eval] = -1.0
    D2 /= dx ** 2; D1 /= (2.0 * dx)
    sol = solve_ivp(lambda _, u: nu_test * (D2 @ u) - u * (D1 @ u), (0.0, t_plot[-1]), u0, t_eval=t_plot, method="BDF", rtol=1e-8, atol=1e-10)
    return sol.y.T
