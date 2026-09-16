from typing import Callable
import numpy as np
import sympy as sp
import scipy.ndimage
from scipy.integrate import solve_ivp

# 2D Allen-Cahn: u_t - kappa*(u_xx+u_yy) - 5u(1-u^2) = 0, homogeneous Dirichlet BC on all 4 faces of
# the square [min_x, max_x]^2, IC u(x,y,0) = sin(x)*sin(pi*y). Note the IC's x-frequency is 1, not pi --
# still a genuine eigenmode of the Laplacian (eigenfunctions on a product domain are separable
# regardless of frequency), so the Padé approximant's linear-diffusion-only recursion stays well-posed;
# only the nonlinear reaction term 5u(1-u^2) is left to the NN correction (same choice as 1D Allen-Cahn
# and Burgers).

def symbolic_allencahn2d() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    x_sym, y_sym = sp.symbols("x y", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    u_initial = sp.sin(x_sym) * sp.sin(sp.pi * y_sym)
    pde_op = lambda u: kappa_sym * (u.diff(x_sym, 2) + u.diff(y_sym, 2))
    return x_sym, y_sym, t_sym, kappa_sym, u_initial, pde_op

# Numeric IC matching symbolic_allencahn2d's u_initial exactly, so the Padé approximant and the
# reference solver's initial condition can never drift out of sync.
def uv_initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.sin(np.pi * y)

# RK45 matrix-free reference solver for u_t = kappa*(u_xx+u_yy) + 5u(1-u^2), zero Dirichlet BC pinned
# directly on the boundary rows/columns of u_t and u0. Unlike heat2d's interior-only FD grid (which
# needs a separate off-grid interpolation step), x_plot/y_plot here include both domain edges, so the
# boundary values are pinned in place rather than excluded.
def solve_ac2d_ref(x_plot: np.ndarray, y_plot: np.ndarray, t_plot: np.ndarray, kappa_test: float, u0: np.ndarray) -> np.ndarray:
    dx = x_plot[1] - x_plot[0]
    nx_dim, ny_dim = len(x_plot), len(y_plot)

    def rhs(_t, u_flat):
        u = u_flat.reshape(nx_dim, ny_dim)
        lap = scipy.ndimage.laplace(u, mode='constant', cval=0.0) / (dx ** 2)
        u_t = kappa_test * lap + 5.0 * u * (1.0 - u ** 2)
        u_t[0, :] = 0.0; u_t[-1, :] = 0.0
        u_t[:, 0] = 0.0; u_t[:, -1] = 0.0
        return u_t.ravel()

    u0 = u0.copy()
    u0[0, :] = 0.0; u0[-1, :] = 0.0
    u0[:, 0] = 0.0; u0[:, -1] = 0.0

    sol = solve_ivp(rhs, (0.0, t_plot[-1]), u0.ravel(), t_eval=t_plot, method='RK45', rtol=1e-5, atol=1e-6)
    return sol.y.T
