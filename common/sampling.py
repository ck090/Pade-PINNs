from typing import Callable
import numpy as np
import jax.numpy as jnp

# Draw a physics parameter uniformly in log-space
def sample_log_kappa(k_lb: float = 1e-2, k_ub: float = 0.5) -> float:
    return float(np.exp(np.random.uniform(np.log(max(k_lb, 1e-7)), np.log(k_ub))))

# Sample collocation points uniformly in x and t, plus left/right boundary points at matching t.
# Assumes a 1D-space x-domain [min_x, max_x] with a left/right boundary pair (e.g. periodic BC).
def sample_residual_and_bc(nx: int, nbc: int, duration: float, min_x: float, max_x: float, starting_point: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X_res = jnp.array(np.column_stack([
        np.random.uniform(min_x, max_x, nx),
        np.random.uniform(starting_point, duration, nx),
    ]))
    t_bc = np.random.uniform(starting_point, duration, nbc)
    X_bc_left = jnp.array(np.column_stack([np.full(nbc, min_x), t_bc]))
    X_bc_right = jnp.array(np.column_stack([np.full(nbc, max_x), t_bc]))
    return X_res, X_bc_left, X_bc_right

# RAR: sample n_candidates points, evaluate residual_fn(X) -> per-point residual (shape (N,) for a scalar
# PDE, (N, k) for a k-component system), and keep the n_select highest-residual-magnitude points as anchor
# collocation points. residual_fn should close over whatever model params/physics parameter it needs.
def select_rar_points(residual_fn: Callable[[jnp.ndarray], jnp.ndarray], n_candidates: int, n_select: int, duration: float, min_x: float, max_x: float, starting_point: float) -> jnp.ndarray:
    X_candidates = jnp.array(np.column_stack([
        np.random.uniform(min_x, max_x, n_candidates),
        np.random.uniform(starting_point, duration, n_candidates),
    ]))
    residuals = residual_fn(X_candidates)
    residual_mag = jnp.sum(residuals.reshape(residuals.shape[0], -1) ** 2, axis=1)
    top_idx = jnp.argsort(-residual_mag)[:n_select]
    return X_candidates[top_idx]
