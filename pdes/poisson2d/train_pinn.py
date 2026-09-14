import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import pickle
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from common.env import configure_jax
from common.nn import Params, init_params, mlp_forward
from common.sampling import sample_log_kappa, sample_residual_and_bc_2d, select_rar_points_2d
from common.plotting import plot_prediction, animate_prediction, plot_loss_history, plot_pinn_error_evolution, best_pade_pinn_kappa
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.poisson2d.config import Poisson2DConfig, make_test_kappas
from pdes.poisson2d.physics import poisson2d_exact

configure_jax(enable_x64=True)

# Sample collocation, boundary (4 faces), and initial-condition points; IC points/values are on top
# of the shared residual+BC sampler since the plain PINN (unlike Padé+PINN) enforces the IC via a loss
# term. Relaxation starts from rest, so U_ic is identically zero.
def prepare_data(nx: int, nbc: int, nic: int, duration: float, min_x: float, max_x: float, starting_point: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X_res, X_bc = sample_residual_and_bc_2d(nx, nbc, duration, min_x, max_x, starting_point)
    x_ic = np.random.uniform(min_x, max_x, size=nic)
    y_ic = np.random.uniform(min_x, max_x, size=nic)
    X_ic = jnp.array(np.column_stack([x_ic, y_ic, np.full(nic, starting_point)]))
    U_ic = jnp.zeros(nic)
    return X_res, X_bc, X_ic, U_ic

# 2D Poisson pseudo-time relaxation: u_t - kappa*(u_xx+u_yy) - f_src(x,y) = 0; network input X is
# (N, 4) = [x, y, t, kappa]
@jax.jit
def poisson2d_eqn(params: Params, X: jnp.ndarray, kappa: float) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def u_fn(xyt: jnp.ndarray) -> jnp.ndarray:
        return pinn(xyt.reshape(1, -1)).squeeze()

    def compute_pde(xyt: jnp.ndarray) -> jnp.ndarray:
        u_t = jax.grad(u_fn)(xyt)[2]
        H = jax.hessian(u_fn)(xyt)
        return u_t - kappa * (H[0, 0] + H[1, 1]) - jnp.sin(jnp.pi * xyt[0]) * jnp.sin(jnp.pi * xyt[1])
    return jax.vmap(compute_pde)(X)

# PINN: single global network conditioned on physics parameter kappa; 1-component output (u)
class PINN:
    def __init__(self, layer: List[int], lr: float, n_chunks: int = 16, weight_floor: float = 0.0) -> None:
        self.params = init_params(layer, zero_last_layer=True)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(self.params)
        self.layer = layer
        self.lr = lr
        self.n_chunks, self.weight_floor = n_chunks, weight_floor

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, params: Params, X: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, X)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc: jnp.ndarray, X_ic: jnp.ndarray, U_ic: jnp.ndarray, kappa: float, causal_eps: jnp.ndarray) -> Tuple[Any, Tuple[float, float, float]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss)(params, X_res, X_bc, X_ic, U_ic, kappa, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc, X_ic, U_ic, kappa, causal_eps)
        return next_opt_state, (loss_res, loss_bc, loss_ic)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, X_ic: jnp.ndarray, U_ic: jnp.ndarray, kappa: float, causal_eps: jnp.ndarray) -> Tuple[float, float, float]:
        def _append(X: jnp.ndarray) -> jnp.ndarray:
            return jnp.concatenate([X, jnp.full((X.shape[0], 1), kappa)], axis=1)
        X_res_full = _append(X_res)
        X_bc_full = _append(X_bc)
        X_ic_full = _append(X_ic)
        # PDE residual, causally weighted
        eq = poisson2d_eqn(params, X_res_full, kappa)
        loss_residual = causal_residual_loss(eq, X_res[:, 2], causal_eps, self.n_chunks, self.weight_floor)
        # Dirichlet BC: u = 0 on all 4 faces
        out_bc = self.forward(params, X_bc_full)
        loss_boundary = jnp.mean(out_bc ** 2)
        # Initial condition: relaxation starts from rest, u(x,y,0) = 0
        out_ic = self.forward(params, X_ic_full).squeeze()
        loss_ic = jnp.mean((out_ic - U_ic) ** 2)
        return loss_residual, loss_boundary, loss_ic

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, X_ic: jnp.ndarray, U_ic: jnp.ndarray, kappa: float, causal_eps: jnp.ndarray) -> float:
        loss_residual, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc, X_ic, U_ic, kappa, causal_eps)
        return loss_residual + loss_bc + loss_ic

    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({'params': self.params, 'layer': self.layer, 'lr': self.lr}, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> 'PINN':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = PINN(data['layer'], data['lr'])
        model.params = data['params']
        print(f"Model loaded from {filepath}")
        return model

# Evaluate the PINN against the exact relaxation solution for one kappa, over the full (x,y,t) eval grid
def evaluate_kappa(model: PINN, kappa_test: float, X_grid: np.ndarray, Y_grid: np.ndarray, t_plot: np.ndarray) -> Dict[str, Any]:
    errors = []
    U_pred_full, U_true_full = [], []
    for t_val in t_plot:
        T_grid = np.full_like(X_grid, t_val)
        XYT_full = jnp.array(np.column_stack([X_grid.ravel(), Y_grid.ravel(), T_grid.ravel(), np.full(X_grid.size, kappa_test)]))
        U_pred_t = np.array(model.forward(model.params, XYT_full)).squeeze(-1)
        U_true_t = poisson2d_exact(X_grid, Y_grid, t_val, kappa_test).ravel()

        norm_true = max(np.linalg.norm(U_true_t), 1e-10)
        errors.append(np.linalg.norm(U_pred_t - U_true_t) / norm_true)
        U_pred_full.append(U_pred_t)
        U_true_full.append(U_true_t)
    U_pred_full, U_true_full = np.array(U_pred_full), np.array(U_true_full)

    norm_g = max(float(np.mean(U_true_full ** 2)), 1e-10)
    mse = float(np.mean((U_pred_full - U_true_full) ** 2) / norm_g)

    return {"U_pred_full": U_pred_full, "U_true_full": U_true_full, "errors": errors, "avg": float(np.mean(errors)), "mse": mse}

if __name__ == "__main__":
    cfg = Poisson2DConfig()
    np.random.seed(cfg.seed)
    test_kappas = make_test_kappas(cfg)
    # Plot the held-out kappa where Padé+PINN did best so the two models' figures are directly comparable
    plot_kappa = best_pade_pinn_kappa("models/pade_pinn_poisson2d_error_metrics.json", test_kappas)

    duration, nx, nbc, nic, nx_eval = cfg.duration, cfg.nx, cfg.nbc, cfg.nic, cfg.nx_eval
    min_x, max_x, starting_point = cfg.min_x, cfg.max_x, cfg.starting_point
    layers, lr, epochs = cfg.layers, cfg.lr, cfg.epochs
    param_lb, param_ub = cfg.param_lb, cfg.param_ub

    causal_eps_max, n_chunks = cfg.causal_eps_max, cfg.causal_n_chunks
    CAUSAL_WARMUP_FRAC, CAUSAL_WEIGHT_FLOOR = cfg.causal_warmup_frac, cfg.causal_weight_floor
    model = PINN(layers, lr, n_chunks=n_chunks, weight_floor=CAUSAL_WEIGHT_FLOOR)

    RAR_EVERY, RAR_POOL, RAR_N_ANCHOR = cfg.rar_every, cfg.rar_pool, cfg.rar_n_anchor
    X_res_anchor = None

    print(f"\n{'='*62}")
    print(f"  PINN training  -  single global network  (2D Poisson, pseudo-time relaxation)")
    print(f"  Epochs={epochs}  lr={lr}")
    print(fr"  kappa in [{param_lb, param_ub}]  log-uniform   plot kappa={plot_kappa:.5f} (Padé+PINN best)")
    print(f"{'='*62}")

    for epoch in range(1, epochs + 1):
        X_res, X_bc, X_ic, U_ic = prepare_data(nx, nbc, nic, duration, min_x, max_x, starting_point)
        kappa = float(sample_log_kappa(param_lb, param_ub))
        kappa_j = jnp.asarray(kappa)

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            def residual_fn(X: jnp.ndarray) -> jnp.ndarray:
                kappa_col = jnp.full((X.shape[0], 1), kappa_j)
                return poisson2d_eqn(model.params, jnp.concatenate([X, kappa_col], axis=1), kappa_j)
            X_res_anchor = select_rar_points_2d(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        start = time.time()
        model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc, X_ic, U_ic, kappa_j, current_causal_eps)
        model.params = model.get_params(model.opt_state)
        end = time.time()

        if epoch % 500 == 0:
            loss_res, loss_bc, loss_ic = losses
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}, {float(loss_ic):.3e}]  kappa={kappa:.5f}  causal_eps={float(current_causal_eps):.3f}  {end-start:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pinn_poisson2d.pkl")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_kappas)} held-out κ values")
    print("=" * 65)
    error_metric = {}

    x_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    y_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    # Excludes t=0: the relative-L2 metric divides by ||u_true(t)||, which is exactly 0 at t=0
    # (relaxation starts from rest) — a degenerate, undefined point for a relative error metric.
    t_plot = np.linspace(starting_point, duration, cfg.nt_eval)[1:]
    X_grid, Y_grid = np.meshgrid(x_plot, y_plot)

    for trial, kappa_test in enumerate(test_kappas):
        print(f"\n  [{trial+1}/{len(test_kappas)}]  κ = {kappa_test:.5f}")

        result = evaluate_kappa(model, kappa_test, X_grid, Y_grid, t_plot)
        avg_err, mse, err_t = result["avg"], result["mse"], result["errors"][-1]

        print(f"\n{'='*62}")
        print(f"{'Metric':<12} {'[PINN]':>12}")
        print(f"{'='*62}")
        print(f"{'Rel-L2@t=T':<12} {err_t:>12.3e}")
        print(f"{'L2':<12} {avg_err:>12.3e}")
        print(f"{'Rel_MSE':<12} {mse:>12.3e}")
        print(f"{'Rel_RMSE':<12} {np.sqrt(mse):>12.3e}")
        print(f"{'='*62}")

        error_metric[kappa_test] = {
            'L2': avg_err,
            'Rel_MSE': mse,
            'Rel_RMSE': float(np.sqrt(mse)),
        }

    json_path = "models/pinn_poisson2d_error_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved to {json_path}")

    plot_result = evaluate_kappa(model, plot_kappa, X_grid, Y_grid, t_plot)
    print(f"\nPlotting held-out κ = {plot_kappa:.5f} (Padé+PINN's best)  —  PINN Rel_MSE there = {plot_result['mse']:.3e}")

    # 1D spatial slice at y ≈ 0.5 to reuse the 1D plotting utilities
    idx_y = np.argmin(np.abs(y_plot - 0.5))
    U_pred_slice = plot_result["U_pred_full"][:, idx_y * nx_eval:(idx_y + 1) * nx_eval]
    U_true_slice = plot_result["U_true_full"][:, idx_y * nx_eval:(idx_y + 1) * nx_eval]
    X_plot, T_plot = np.meshgrid(x_plot, t_plot)

    # Real random boundary samples (same distribution/code path used during training), left/right faces only
    _, X_bc_sample = sample_residual_and_bc_2d(1, cfg.nbc_plot, duration, min_x, max_x, starting_point)
    X_bc_sample = np.array(X_bc_sample)
    X_bc_left = X_bc_sample[:cfg.nbc_plot, [0, 2]]
    X_bc_right = X_bc_sample[cfg.nbc_plot:2 * cfg.nbc_plot, [0, 2]]

    plot_pinn_error_evolution(t_plot, plot_result["errors"], {'\\kappa': plot_kappa}, name="poisson2dPINN", folder="pinn")
    plot_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, X_bc_left, X_bc_right, {'\\kappa': plot_kappa}, intervals=[0.2, duration / 2, duration - 0.2], name="poisson2dPINN", folder="pinn")
    animate_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, {'\\kappa': plot_kappa}, name="poisson2dPINN", folder="pinn")

    avg_L2 = float(np.mean([v['L2'] for v in error_metric.values()]))
    avg_mse = float(np.mean([v['Rel_MSE'] for v in error_metric.values()]))
    avg_rmse = float(np.mean([v['Rel_RMSE'] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics — across {len(error_metric)} held-out κ values")
    print("=" * 65)
    print(f"{'Metric':<12} {'[PINN]':>12}")
    print("=" * 62)
    print(f"{'Avg L2':<12} {avg_L2:>12.3e}")
    print(f"{'Rel_MSE':<12} {avg_mse:>12.3e}")
    print(f"{'Rel_RMSE':<12} {avg_rmse:>12.3e}")
    print("=" * 62)
