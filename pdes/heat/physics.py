from typing import Callable
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# 1D heat equation: u_t - kappa*u_xx = 0, periodic BC on [min_x, max_x], IC u(x,0) = cos(2*pi*x).

def symbolic_heat() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    x_sym = sp.Symbol("x", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    u_initial = sp.cos(2 * sp.pi * x_sym)
    pde_op = lambda u: kappa_sym * u.diff(x_sym, 2)
    return x_sym, t_sym, kappa_sym, u_initial, pde_op

# Numeric IC matching symbolic_heat's u_initial exactly (same expression, lambdified) so the Padé
# approximant and the reference solver's initial condition can never drift out of sync.
def uv_initial_condition(x: np.ndarray) -> np.ndarray:
    return np.cos(2 * np.pi * x)

# RK45 reference solution: central finite difference with periodic BCs
def solve_heat_ref(x_plot: np.ndarray, t_plot: np.ndarray, kappa_test: float, u0: np.ndarray) -> np.ndarray:
    nx_eval = len(x_plot)
    dx = x_plot[1] - x_plot[0]

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        return kappa_test * (np.roll(u, 1) - 2 * u + np.roll(u, -1)) / dx ** 2

    # x_plot's last point duplicates the first under periodicity; integrate on the distinct points, mirror it back
    sol = solve_ivp(rhs, (0.0, t_plot[-1]), u0[:-1], t_eval=t_plot, method="RK45", rtol=1e-6, atol=1e-9)
    return np.concatenate([sol.y.T, sol.y.T[:, :1]], axis=1)
