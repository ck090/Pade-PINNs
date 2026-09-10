from typing import Callable
import numpy as np
import sympy as sp

# Fokker-Planck equation for the Ornstein-Uhlenbeck SDE dX = -THETA*X dt + kappa*dW:
#   p_t = THETA*d/dx(x*p) + (kappa^2/2)*p_xx,  i.e.  p_t - THETA*(p + x*p_x) - (kappa^2/2)*p_xx = 0
# with p -> 0 at both ends of the domain (Dirichlet) and a Gaussian initial density N(X0_INIT, S0_INIT^2).
THETA = 0.5
X0_INIT, S0_INIT = 2.0, 1.5

def initial_density(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x - X0_INIT) ** 2 / (2 * S0_INIT ** 2)) / np.sqrt(2 * np.pi * S0_INIT ** 2)

# Closed-form OU density: Gaussian for all t, mean and variance relaxing to (0, kappa^2/(2*THETA)); the ground truth for both models, no solver needed
def fokker_planck_exact(x: np.ndarray, t: np.ndarray, kappa: float) -> np.ndarray:
    mean = X0_INIT * np.exp(-THETA * t)
    var_inf = kappa ** 2 / (2 * THETA)
    var = var_inf + (S0_INIT ** 2 - var_inf) * np.exp(-2 * THETA * t)
    return np.exp(-(x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

# p_t = pde_op(p). The operator is already linear, so unlike Schrodinger nothing is dropped: the Padé baseline carries
# the full PDE and the PINN correction only fixes its truncation error in t. x_sym/t_sym/kappa_sym are fresh symbols per call.
def symbolic_fokker_planck() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    x_sym = sp.Symbol("x", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    # Same density as initial_density -- a Padé built for a different IC than the exact solution fails silently
    x0, s0, theta = sp.Rational(X0_INIT), sp.Rational(S0_INIT), sp.Rational(THETA)
    p_initial = sp.exp(-(x_sym - x0) ** 2 / (2 * s0 ** 2)) / sp.sqrt(2 * sp.pi * s0 ** 2)
    pde_op = lambda p: theta * (x_sym * p).diff(x_sym) + kappa_sym ** 2 / 2 * p.diff(x_sym, 2)
    return x_sym, t_sym, kappa_sym, p_initial, pde_op
