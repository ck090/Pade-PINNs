from typing import Callable
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# Nonlinear Schrodinger equation: i*w_t + kappa*w_xx + |w|^2*w = 0 for w = u + iv,
# split into real/imaginary components:
#   u_t + kappa*v_xx + (u^2+v^2)*v = 0
#   v_t - kappa*u_xx - (u^2+v^2)*u = 0
#
# kappa scales the DISPERSION, not a potential. A potential term (-kappa*w) is a pure gauge:
# w = exp(-i*kappa*t)*psi removes it exactly, leaving |w|^2 independent of kappa, so every metric
# built on H = u^2+v^2 is blind to it. On the dispersion kappa sets the oscillation rate of mode k
# (frequency ~ kappa*k^2) and is genuinely visible in H. The spatial rescaling xi = x/sqrt(kappa)
# that would remove it is blocked by the fixed domain and fixed-wavenumber IC below.

# Initial condition: a single Fourier eigenmode of L = kappa*d_xx on the periodic domain.
# The eigenmode is load-bearing, not cosmetic. The time-Pade coefficients are formed pointwise as
# ratios of the Taylor coefficients (b2 = L^4 u0 / (12 L^2 u0)); for an eigenfunction that collapses
# to the positive constant lambda^2/12, so the denominator 1 + b2*t^2 can never vanish. For anything
# that is NOT an eigenfunction -- sech, a gaussian, or even a sum of two modes -- the ratio varies
# with x, changes sign at the nodes of L^2 u0, and plants real poles inside t in [0, 2] that blow the
# Pade baseline up by ~10^4. Amplitude stays small because the |w|^2*w term the Pade omits scales as A^3.
# DOMAIN_LENGTH must equal cfg.max_x - cfg.min_x, or the IC is not periodic on the solved domain.
DOMAIN_LENGTH = 10.0
IC_MODE = 1
IC_AMPLITUDE = 0.5

def uv_initial_condition_mode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = 2.0 * np.pi * IC_MODE / DOMAIN_LENGTH
    return IC_AMPLITUDE * np.cos(k * x), np.zeros_like(x)

# The linear part of the PDE above (drop the |w|^2*w nonlinearity): u_t = pde_op_u(v), v_t = pde_op_v(u).
# This is what the Padé-in-time approximant is built from; the PINN correction is left to capture
# the nonlinearity. x_sym/t_sym/kappa_sym are fresh symbols for each call.
def symbolic_linearized_schrodinger() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, sp.Expr, Callable, Callable]:
    x_sym = sp.Symbol("x", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    # Same mode as uv_initial_condition_mode -- both derive from the constants above, since a Pade
    # built for a different IC than the reference solver is seeded with fails silently.
    k_sym = 2 * sp.pi * IC_MODE / DOMAIN_LENGTH
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
