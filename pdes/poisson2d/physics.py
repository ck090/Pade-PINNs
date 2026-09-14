from typing import Callable
import numpy as np
import sympy as sp

# 2D Poisson via pseudo-time relaxation: u_t = kappa*(u_xx+u_yy) + f_src(x,y); the steady state (t ->
# infinity) solves -kappa*Lap(u) = f_src. Homogeneous Dirichlet BC on all 4 faces of the square
# [min_x, max_x]^2, forcing f_src = sin(pi*x)*sin(pi*y) (the fundamental eigenmode of the
# zero-Dirichlet Laplacian on that square), relaxation starts from rest: u(x,y,0) = 0.

def symbolic_poisson2d() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, sp.Expr, Callable]:
    x_sym, y_sym = sp.symbols("x y", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    u_initial = sp.Integer(0)
    f_src_expr = sp.sin(sp.pi * x_sym) * sp.sin(sp.pi * y_sym)
    pde_op = lambda u: kappa_sym * (u.diff(x_sym, 2) + u.diff(y_sym, 2))
    return x_sym, y_sym, t_sym, kappa_sym, u_initial, f_src_expr, pde_op

# Numeric forcing term matching symbolic_poisson2d's f_src_expr exactly, so the Padé approximant and
# the exact-solution checks below can never drift out of sync.
def f_src_numeric(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Exact single-mode relaxation solution: a(t) = (exp(mu*t)-1)/mu, mu = -2*pi^2*kappa; a(t) -> -1/mu as
# t -> infinity, at which point this reduces to poisson2d_steady_exact below.
def poisson2d_exact(x: np.ndarray, y: np.ndarray, t: float, kappa: float) -> np.ndarray:
    mu = -2.0 * np.pi ** 2 * kappa
    a_t = (np.exp(mu * t) - 1.0) / mu
    return a_t * f_src_numeric(x, y)

# Exact steady-state solution of -kappa*Lap(u) = f_src (the t -> infinity limit of the relaxation)
def poisson2d_steady_exact(x: np.ndarray, y: np.ndarray, kappa: float) -> np.ndarray:
    return (1.0 / (2.0 * np.pi ** 2 * kappa)) * f_src_numeric(x, y)
