from typing import Callable
import numpy as np
import sympy as sp

# 2D isotropic Ornstein-Uhlenbeck Fokker-Planck: dX_i = -theta*X_i dt + kappa*dW_i, independently for
# i=x,y (isotropic drift/diffusion, no cross terms). theta (mean-reversion rate) and kappa (noise
# amplitude) are BOTH swept parameters here, unlike the 1D package (theta fixed at 0.5). Because the
# operator separates completely across axes, both the exact solution and the Padé baseline are
# literal products of two copies of the already-validated 1D Ornstein-Uhlenbeck formula/approximant
# -- one per axis, sharing the same (theta, kappa) -- rather than a new 2D derivation.
# Dirichlet p=0 on all 4 edges of the square [min_x, max_x]^2, Gaussian initial density (product of
# two 1D Gaussians, isotropic).
X0_INIT, S0_INIT = 2.0, 1.5

def initial_density_1d(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x - X0_INIT) ** 2 / (2 * S0_INIT ** 2)) / np.sqrt(2 * np.pi * S0_INIT ** 2)

def initial_density_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return initial_density_1d(x) * initial_density_1d(y)

# Closed-form 1D OU density: Gaussian for all t, mean and variance relaxing to (0, kappa^2/(2*theta))
def fokker_planck_exact_1d(x: np.ndarray, t: np.ndarray, theta: float, kappa: float) -> np.ndarray:
    mean = X0_INIT * np.exp(-theta * t)
    var_inf = kappa ** 2 / (2 * theta)
    var = var_inf + (S0_INIT ** 2 - var_inf) * np.exp(-2 * theta * t)
    return np.exp(-(x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

# 2D density: exact product of two independent 1D OU densities (isotropic, separable system) -- the
# ground truth evaluate_theta_kappa uses, so it always matches whatever the 1D IC/dynamics are
def fokker_planck_exact_2d(x: np.ndarray, y: np.ndarray, t: np.ndarray, theta: float, kappa: float) -> np.ndarray:
    return fokker_planck_exact_1d(x, t, theta, kappa) * fokker_planck_exact_1d(y, t, theta, kappa)

# Symbolic 1D OU building block, shared across the x, y axes (same theta, kappa, IC shape). The Padé
# baseline for the full 2D problem is the product of two lambdified copies of this one approximant
# (one call per axis, at evaluation time) -- not a new 2D symbolic derivation.
def symbolic_fokker_planck_1d() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    xi_sym = sp.Symbol("xi", real=True)
    t_sym = sp.Symbol("t", real=True, positive=True)
    theta_sym, kappa_sym = sp.symbols("theta kappa", real=True, positive=True)
    x0, s0 = sp.Rational(X0_INIT), sp.Rational(S0_INIT)
    p_initial = sp.exp(-(xi_sym - x0) ** 2 / (2 * s0 ** 2)) / sp.sqrt(2 * sp.pi * s0 ** 2)
    pde_op = lambda p: theta_sym * (xi_sym * p).diff(xi_sym) + kappa_sym ** 2 / 2 * p.diff(xi_sym, 2)
    return xi_sym, t_sym, theta_sym, kappa_sym, p_initial, pde_op
