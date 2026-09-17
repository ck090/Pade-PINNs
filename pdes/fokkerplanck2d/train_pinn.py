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
from common.plotting import plot_prediction, plot_pinn_error_evolution
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.fokkerplanck2d.config import FokkerPlanck2DConfig, make_test_params
from pdes.fokkerplanck2d.physics import X0_INIT, S0_INIT, initial_density_2d, fokker_planck_exact_2d

configure_jax(enable_x64=True)

"""Sample collocation, boundary, and initial-condition points; IC points/values are on top of the shared residual+BC sampler since the plain PINN (unlike Padé+PINN) enforces the IC via a loss term"""
def prepare_data(nx: int, nbc: int, nic: int, duration: float, min_x: float, max_x: float, starting_point: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X_res, X_bc = sample_residual_and_bc_2d(nx, nbc, duration, min_x, max_x, starting_point)
    x_ic = np.random.uniform(min_x, max_x, size=nic)
    y_ic = np.random.uniform(min_x, max_x, size=nic)
    X_ic = jnp.array(np.column_stack([x_ic, y_ic, np.full(nic, starting_point)]))
    P_ic = jnp.array(initial_density_2d(x_ic, y_ic))
    return X_res, X_bc, X_ic, P_ic

"""2D Fokker-Planck PDE: p_t - theta*(2p + x*p_x + y*p_y) - (kappa^2/2)*(p_xx+p_yy) = 0;
network input X is (N, 5) = [x, y, t, theta, kappa], derivatives w.r.t. indices 0,1 (x,y) and 2 (t)"""
@jax.jit
def fp2d_eqn(params: Params, X: jnp.ndarray, theta: float, kappa: float) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def p_fn(xyt: jnp.ndarray) -> jnp.ndarray:
        return pinn(xyt.reshape(1, -1)).squeeze()

    def compute_pde(xyt: jnp.ndarray) -> jnp.ndarray:
        x, y = xyt[0], xyt[1]
        dp = jax.grad(p_fn)(xyt)
        H = jax.hessian(p_fn)(xyt)
        drift = theta * (2 * p_fn(xyt) + x * dp[0] + y * dp[1])
        diffusion = 0.5 * kappa ** 2 * (H[0, 0] + H[1, 1])
        return dp[2] - drift - diffusion
    return jax.vmap(compute_pde)(X)

"""PINN: single global network conditioned on physics parameters (theta, kappa); 1-component output (p)"""
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
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc: jnp.ndarray, X_ic: jnp.ndarray, P_ic: jnp.ndarray, theta: float, kappa: float, causal_eps: jnp.ndarray) -> Tuple[Any, Tuple[float, float, float]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss)(params, X_res, X_bc, X_ic, P_ic, theta, kappa, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc, X_ic, P_ic, theta, kappa, causal_eps)
        return next_opt_state, (loss_res, loss_bc, loss_ic)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, X_ic: jnp.ndarray, P_ic: jnp.ndarray, theta: float, kappa: float, causal_eps: jnp.ndarray) -> Tuple[float, float, float]:
        N_res, N_bc, N_ic = X_res.shape[0], X_bc.shape[0], X_ic.shape[0]
        def _append(X: jnp.ndarray, n: int) -> jnp.ndarray:
            return jnp.concatenate([X, jnp.full((n, 1), theta), jnp.full((n, 1), kappa)], axis=1)
        X_res_full = _append(X_res, N_res)
        X_bc_full = _append(X_bc, N_bc)
        X_ic_full = _append(X_ic, N_ic)
        # PDE residual, causally weighted; the residual is already a single scalar per point
        eq = fp2d_eqn(params, X_res_full, theta, kappa)
        loss_residual = causal_residual_loss(eq, X_res[:, 2], causal_eps, self.n_chunks, self.weight_floor)
        # Dirichlet BC: the density vanishes on all 4 edges of the square
        out_bc = self.forward(params, X_bc_full)
        loss_boundary = jnp.mean(out_bc ** 2)
        # Initial condition
        out_ic = self.forward(params, X_ic_full).squeeze()
        loss_ic = jnp.mean((out_ic - P_ic) ** 2)
        return loss_residual, loss_boundary, loss_ic

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, X_ic: jnp.ndarray, P_ic: jnp.ndarray, theta: float, kappa: float, causal_eps: jnp.ndarray) -> float:
        loss_residual, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc, X_ic, P_ic, theta, kappa, causal_eps)
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

"""Evaluate the PINN against the exact OU density for one (theta, kappa) pair, over the full (x,y,t) eval grid"""
def evaluate_theta_kappa(model: PINN, theta_test: float, kappa_test: float, X_grid: np.ndarray, Y_grid: np.ndarray, t_plot: np.ndarray) -> Dict[str, Any]:
    shape = X_grid.shape
    x_ax, y_ax = X_grid[:, 0], Y_grid[0, :]

    errors, mass, min_p = [], [], []
    mse_terms, true_sq_terms = [], []
    for t_val in t_plot:
        T_grid = np.full_like(X_grid, t_val)
        XYT_full = jnp.array(np.column_stack([X_grid.ravel(), Y_grid.ravel(), T_grid.ravel(),
                                               np.full(X_grid.size, theta_test), np.full(X_grid.size, kappa_test)]))
        U_pred_t = np.array(model.forward(model.params, XYT_full)).squeeze(-1)
        U_true_t = fokker_planck_exact_2d(X_grid, Y_grid, t_val, theta_test, kappa_test).ravel()

        norm_true = max(np.linalg.norm(U_true_t), 1e-10)
        errors.append(np.linalg.norm(U_pred_t - U_true_t) / norm_true)
        mse_terms.append(np.mean((U_pred_t - U_true_t) ** 2))
        true_sq_terms.append(np.mean(U_true_t ** 2))
        mass.append(np.trapezoid(np.trapezoid(U_pred_t.reshape(shape), y_ax, axis=1), x_ax, axis=0))
        min_p.append(float(U_pred_t.min()))

    norm_g = max(float(np.mean(true_sq_terms)), 1e-10)
    mse = float(np.mean(mse_terms) / norm_g)

    return {
        "errors": errors, "avg": float(np.mean(errors)), "mse": mse,
        "mass_err": float(np.max(np.abs(np.array(mass) - 1.0))), "min_p": float(np.min(min_p)),
    }

"""Main"""
if __name__ == "__main__":
    cfg = FokkerPlanck2DConfig()
    np.random.seed(cfg.seed)
    test_thetas, test_kappas = make_test_params(cfg)

    # Plot the held-out (theta, kappa) where Padé+PINN did best, so the two models' figures are directly comparable
    pade_metrics_path = "models/pade_pinn_fokkerplanck2d_error_metrics.json"
    if not os.path.exists(pade_metrics_path):
        raise FileNotFoundError(f"{pade_metrics_path} not found -- run train_pade_pinn first; the PINN plots at Padé+PINN's best (theta, kappa)")
    with open(pade_metrics_path) as f:
        pade_metrics = json.load(f)
    best_pade_key = min(pade_metrics, key=lambda k: pade_metrics[k]["Rel_MSE"][1])
    plot_theta, plot_kappa = pade_metrics[best_pade_key]["theta"], pade_metrics[best_pade_key]["kappa"]

    duration, nx, nbc, nic, nx_eval, nt_eval = cfg.duration, cfg.nx, cfg.nbc, cfg.nic, cfg.nx_eval, cfg.nt_eval
    min_x, max_x, starting_point = cfg.min_x, cfg.max_x, cfg.starting_point
    layers, lr, epochs = cfg.layers, cfg.lr, cfg.epochs
    theta_lb, theta_ub, kappa_lb, kappa_ub = cfg.theta_lb, cfg.theta_ub, cfg.kappa_lb, cfg.kappa_ub

    causal_eps_max, n_chunks = cfg.causal_eps_max, cfg.causal_n_chunks
    CAUSAL_WARMUP_FRAC, CAUSAL_WEIGHT_FLOOR = cfg.causal_warmup_frac, cfg.causal_weight_floor
    model = PINN(layers, lr, n_chunks=n_chunks, weight_floor=CAUSAL_WEIGHT_FLOOR)

    RAR_EVERY, RAR_POOL, RAR_N_ANCHOR = cfg.rar_every, cfg.rar_pool, cfg.rar_n_anchor
    X_res_anchor = None

    print(f"\n{'='*62}")
    print(f"  PINN training  -  single global network  (2D Fokker-Planck, isotropic OU)")
    print(f"  Epochs={epochs}  lr={lr}")
    print(f"  time={duration}  nx={nx}  min_x={min_x} max_x={max_x}")
    print(f"  x0={X0_INIT}  s0={S0_INIT}")
    print(f"  theta in [{theta_lb}, {theta_ub}]  kappa in [{kappa_lb}, {kappa_ub}]  log-uniform   plot (theta,kappa)=({plot_theta:.5f},{plot_kappa:.5f}) (Padé+PINN best)")
    print(f"{'='*62}")

    for epoch in range(1, epochs + 1):
        X_res, X_bc, X_ic, P_ic = prepare_data(nx, nbc, nic, duration, min_x, max_x, starting_point)
        theta = float(sample_log_kappa(theta_lb, theta_ub))
        kappa = float(sample_log_kappa(kappa_lb, kappa_ub))
        theta_j, kappa_j = jnp.asarray(theta), jnp.asarray(kappa)

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            def residual_fn(X: jnp.ndarray) -> jnp.ndarray:
                pk_col = jnp.column_stack([jnp.full(X.shape[0], theta_j), jnp.full(X.shape[0], kappa_j)])
                return fp2d_eqn(model.params, jnp.concatenate([X, pk_col], axis=1), theta_j, kappa_j)
            X_res_anchor = select_rar_points_2d(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        start = time.time()
        model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc, X_ic, P_ic, theta_j, kappa_j, current_causal_eps)
        model.params = model.get_params(model.opt_state)
        end = time.time()

        if epoch % 500 == 0:
            loss_res, loss_bc, loss_ic = losses
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}, {float(loss_ic):.3e}]  theta={theta:.4f}  kappa={kappa:.4f}  causal_eps={float(current_causal_eps):.3f}  {end-start:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pinn_fokkerplanck2d.pkl")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_thetas)} held-out (θ, κ) pairs")
    print("=" * 65)
    error_metric = {}

    x_plot = np.linspace(min_x, max_x, nx_eval)
    y_plot = np.linspace(min_x, max_x, nx_eval)
    t_plot = np.linspace(starting_point, duration, nt_eval)
    X_grid, Y_grid = np.meshgrid(x_plot, y_plot, indexing="ij")

    for trial, (theta_test, kappa_test) in enumerate(zip(test_thetas, test_kappas)):
        print(f"\n  [{trial+1}/{len(test_thetas)}]  θ = {theta_test:.5f}  κ = {kappa_test:.5f}")

        result = evaluate_theta_kappa(model, theta_test, kappa_test, X_grid, Y_grid, t_plot)
        avg_err, mse, err_t = result["avg"], result["mse"], result["errors"][-1]

        print(f"\n{'='*62}")
        print(f"{'Metric':<12} {'[PINN]':>12}")
        print(f"{'='*62}")
        print(f"{'Rel-L2@t=T':<12} {err_t:>12.3e}")
        print(f"{'L2':<12} {avg_err:>12.3e}")
        print(f"{'Rel_MSE':<12} {mse:>12.3e}")
        print(f"{'Rel_RMSE':<12} {np.sqrt(mse):>12.3e}")
        print(f"{'Mass err':<12} {result['mass_err']:>12.3e}")
        print(f"{'Min p':<12} {result['min_p']:>12.3e}")
        print(f"{'='*62}")

        key = f"{theta_test:.6f}_{kappa_test:.6f}"
        error_metric[key] = {
            'theta': float(theta_test), 'kappa': float(kappa_test),
            'L2': avg_err, 'Rel_MSE': mse, 'Rel_RMSE': float(np.sqrt(mse)),
            'Mass_err': result['mass_err'], 'Min_p': result['min_p'],
        }

    json_path = "models/pinn_fokkerplanck2d_error_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved to {json_path}")

    plot_result = evaluate_theta_kappa(model, plot_theta, plot_kappa, X_grid, Y_grid, t_plot)
    print(f"\nPlotting held-out (θ,κ) = ({plot_theta:.5f},{plot_kappa:.5f}) (Padé+PINN's best)  —  PINN Rel_MSE there = {plot_result['mse']:.3e}")

    # (x,t) slice through y=X0_INIT (the IC's peak), matching the same slice train_pade_pinn plots
    x_slice = np.linspace(min_x, max_x, 512)
    t_slice = np.linspace(starting_point, duration, 512)
    X_plot, T_plot = np.meshgrid(x_slice, t_slice)
    Y_plot = np.full_like(X_plot, X0_INIT)
    XYT_slice = jnp.array(np.column_stack([X_plot.ravel(), Y_plot.ravel(), T_plot.ravel(),
                                            np.full(X_plot.size, plot_theta), np.full(X_plot.size, plot_kappa)]))
    U_pred_slice = np.array(model.forward(model.params, XYT_slice)).reshape(512, 512)
    U_true_slice = fokker_planck_exact_2d(X_plot, Y_plot, T_plot, plot_theta, plot_kappa)

    # Real boundary samples (same distribution/code path used during training); columns [0, 2] = (x, t), y dropped
    _, X_bc_sample = sample_residual_and_bc_2d(1, cfg.nbc_plot, duration, min_x, max_x, starting_point)
    X_bc_sample = np.array(X_bc_sample)
    X_bc_left = X_bc_sample[:cfg.nbc_plot, [0, 2]]
    X_bc_right = X_bc_sample[cfg.nbc_plot:2 * cfg.nbc_plot, [0, 2]]

    plot_pinn_error_evolution(t_plot, plot_result["errors"], {'\\theta': plot_theta, '\\kappa': plot_kappa}, name="fokkerplanck2dPINN", folder="pinn")
    plot_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, X_bc_left, X_bc_right, {'\\theta': plot_theta, '\\kappa': plot_kappa}, intervals=[0.1, duration / 2, duration - 0.05], name="fokkerplanck2dPINN", folder="pinn")

    avg_L2 = float(np.mean([v['L2'] for v in error_metric.values()]))
    avg_mse = float(np.mean([v['Rel_MSE'] for v in error_metric.values()]))
    avg_rmse = float(np.mean([v['Rel_RMSE'] for v in error_metric.values()]))
    avg_mass = float(np.mean([v['Mass_err'] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics — across {len(error_metric)} held-out (θ, κ) pairs")
    print("=" * 65)
    print(f"{'Metric':<12} {'[PINN]':>12}")
    print("=" * 62)
    print(f"{'Avg L2':<12} {avg_L2:>12.3e}")
    print(f"{'Rel_MSE':<12} {avg_mse:>12.3e}")
    print(f"{'Rel_RMSE':<12} {avg_rmse:>12.3e}")
    print(f"{'Mass err':<12} {avg_mass:>12.3e}")
    print("=" * 62)
