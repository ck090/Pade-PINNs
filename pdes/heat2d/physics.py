from typing import Any, Callable
import numpy as np
import sympy as sp
from scipy.sparse import diags, identity, kron
from scipy.integrate import solve_ivp

# 2D heat equation: u_t - kappa*(u_xx + u_yy) = 0, homogeneous Dirichlet BC on all 4 faces of the
# square [min_x, max_x]^2, IC u(x,y,0) = sin(pi*x)*sin(pi*y).

def symbolic_heat2d() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Expr, Callable]:
    x_sym, y_sym = sp.symbols("x y", real=True)
    t_sym, kappa_sym = sp.symbols("t kappa", real=True, positive=True)
    u_initial = sp.sin(sp.pi * x_sym) * sp.sin(sp.pi * y_sym)
    pde_op = lambda u: kappa_sym * (u.diff(x_sym, 2) + u.diff(y_sym, 2))
    return x_sym, y_sym, t_sym, kappa_sym, u_initial, pde_op

# Numeric IC matching symbolic_heat2d's u_initial exactly, so the Padé approximant and the reference
# solver's initial condition can never drift out of sync.
def uv_initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# 2D discrete Laplacian (5-point stencil) on an n x n interior grid; homogeneous Dirichlet BC is
# implicit in the excluded boundary rows
def build_laplacian_2d(n: int, h: float) -> Any:
    main, off = -2.0 * np.ones(n), np.ones(n - 1)
    D2 = diags([off, main, off], [-1, 0, 1]) / h ** 2
    I = identity(n, format="csr")
    return (kron(I, D2) + kron(D2, I)).tocsr()

# Numerically integrate u_t = kappa * Laplacian(u), Dirichlet BC=0, from u0_grid -- the ground truth
# evaluate_kappa uses, so it always matches whatever u_initial is
def solve_heat2d_reference(u0_grid: np.ndarray, kappa: float, L: Any, t_eval: np.ndarray) -> np.ndarray:
    n = u0_grid.shape[0]
    sol = solve_ivp(lambda t, u: kappa * (L @ u), (t_eval[0], t_eval[-1]), u0_grid.ravel(),
                     t_eval=t_eval, method="BDF", jac=lambda t, u: kappa * L, rtol=1e-6, atol=1e-9)
    return sol.y.T.reshape(len(t_eval), n, n)
