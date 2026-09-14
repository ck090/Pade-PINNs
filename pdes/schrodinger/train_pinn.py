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
from common.sampling import sample_log_kappa, sample_residual_and_bc, select_rar_points
from common.plotting import plot_prediction, animate_prediction, plot_loss_history, plot_pinn_error_evolution, best_pade_pinn_kappa
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.schrodinger.config import SchrodingerConfig, make_test_kappas
from pdes.schrodinger.physics import uv_initial_condition_mode, solve_schrodinger_ref

configure_jax(enable_x64=True)

"""Sample collocation, boundary, and initial-condition points; IC points/values are on top of the shared residual+BC sampler since the plain PINN (unlike Padé+PINN) enforces the IC via a loss term"""
def prepare_data(nx: int, nbc: int, nic: int, duration: float, min_x: float, max_x: float, starting_point: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X_res, X_bc_left, X_bc_right = sample_residual_and_bc(nx, nbc, duration, min_x, max_x, starting_point)
    x_ic = np.random.uniform(min_x, max_x, size=nic)
    X_ic = jnp.array([[x, starting_point] for x in x_ic])
    u0_vals, v0_vals = uv_initial_condition_mode(x_ic)
    UV_ic = jnp.array(np.stack([u0_vals, v0_vals], axis=1))
    return X_res, X_bc_left, X_bc_right, X_ic, UV_ic

"""Nonlinear Schrodinger PDE: i*w_t + kappa*w_xx + |w|^2*w = 0 for w = u + iv, split into real/imaginary components: u_t + kappa*v_xx + (u^2+v^2)*v = 0, v_t - kappa*u_xx - (u^2+v^2)*u = 0"""
@jax.jit
def schrodinger_eqn(params: Params, X: jnp.ndarray, kappa: float) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def u_single_real(xt: jnp.ndarray) -> jnp.ndarray:
        return pinn(xt.reshape(1, -1))[:, 0].squeeze()
    def u_single_imag(xt: jnp.ndarray) -> jnp.ndarray:
        return pinn(xt.reshape(1, -1))[:, 1].squeeze()

    def compute_pde(xt: jnp.ndarray) -> jnp.ndarray:
        u = u_single_real(xt)
        v = u_single_imag(xt)
        u_t = jax.grad(u_single_real)(xt)[1]
        v_t = jax.grad(u_single_imag)(xt)[1]
        u_xx = jax.hessian(u_single_real)(xt)[0, 0]
        v_xx = jax.hessian(u_single_imag)(xt)[0, 0]
        h2 = u ** 2 + v ** 2
        pde_real = u_t + kappa * v_xx + h2 * v
        pde_imag = v_t - kappa * u_xx - h2 * u
        return jnp.array([pde_real, pde_imag])
    return jax.vmap(compute_pde)(X)

"""PINN: single global network conditioned on physics parameter kappa; 2-component output (u, v)"""
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
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, X_ic: jnp.ndarray, UV_ic: jnp.ndarray, kappa: float, causal_eps: jnp.ndarray) -> Tuple[Any, Tuple[float, float, float]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss)(params, X_res, X_bc_left, X_bc_right, X_ic, UV_ic, kappa, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc_left, X_bc_right, X_ic, UV_ic, kappa, causal_eps)
        return next_opt_state, (loss_res, loss_bc, loss_ic)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, X_ic: jnp.ndarray, UV_ic: jnp.ndarray, kappa: float, causal_eps: jnp.ndarray) -> Tuple[float, float, float]:
        N_res = X_res.shape[0]
        N_bc = X_bc_left.shape[0]
        N_ic = X_ic.shape[0]
        def _append(X: jnp.ndarray, n: int) -> jnp.ndarray:
            return jnp.concatenate([X, jnp.full((n, 1), kappa)], axis=1)
        X_res_full = _append(X_res, N_res)
        X_bcl_full = _append(X_bc_left, N_bc)
        X_bcr_full = _append(X_bc_right, N_bc)
        X_ic_full = _append(X_ic, N_ic)
        # PDE residual, causally weighted: fold (u, v) into one flat array (with a matching
        # duplicated t) so the shared causal_residual_loss, written for a scalar residual, needs no
        # change for this 2-component PDE
        eq = schrodinger_eqn(params, X_res_full, kappa)
        combined = jnp.concatenate([eq[:, 0], eq[:, 1]])
        t_dup = jnp.concatenate([X_res[:, 1], X_res[:, 1]])
        loss_residual = causal_residual_loss(combined, t_dup, causal_eps, self.n_chunks, self.weight_floor)
        # Periodic BC: u(left)=u(right), v(left)=v(right) and derivatives
        out_bcl = self.forward(params, X_bcl_full)
        out_bcr = self.forward(params, X_bcr_full)
        ul_pred, vl_pred = out_bcl[:, 0], out_bcl[:, 1]
        ur_pred, vr_pred = out_bcr[:, 0], out_bcr[:, 1]
        loss_boundary = jnp.mean((ul_pred - ur_pred) ** 2 + (vl_pred - vr_pred) ** 2)
        def u_real_fn(xt: jnp.ndarray) -> jnp.ndarray:
            return self.forward(params, xt.reshape(1, -1))[:, 0].squeeze()
        def u_imag_fn(xt: jnp.ndarray) -> jnp.ndarray:
            return self.forward(params, xt.reshape(1, -1))[:, 1].squeeze()
        grad_u = jax.grad(u_real_fn)
        grad_v = jax.grad(u_imag_fn)
        u_x_left = jax.vmap(lambda xt: grad_u(xt)[0])(X_bcl_full)
        u_x_right = jax.vmap(lambda xt: grad_u(xt)[0])(X_bcr_full)
        v_x_left = jax.vmap(lambda xt: grad_v(xt)[0])(X_bcl_full)
        v_x_right = jax.vmap(lambda xt: grad_v(xt)[0])(X_bcr_full)
        loss_boundary += jnp.mean((u_x_left - u_x_right) ** 2 + (v_x_left - v_x_right) ** 2)
        # Initial condition
        out_ic = self.forward(params, X_ic_full)
        loss_ic = jnp.mean((out_ic - UV_ic) ** 2)
        return loss_residual, loss_boundary, loss_ic

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, X_ic: jnp.ndarray, UV_ic: jnp.ndarray, kappa: float, causal_eps: jnp.ndarray) -> float:
        loss_residual, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc_left, X_bc_right, X_ic, UV_ic, kappa, causal_eps)
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

"""Evaluate the PINN against the reference solver for one kappa, over the full (x,t) eval grid"""
def evaluate_kappa(model: PINN, kappa_test: float, x_plot: np.ndarray, t_plot: np.ndarray, XT_flat: np.ndarray) -> Dict[str, Any]:
    nx_eval, nt_eval = len(x_plot), len(t_plot)
    XT_full = jnp.array(np.column_stack([XT_flat, np.full(XT_flat.shape[0], kappa_test)]))
    UV_pred = np.array(model.forward(model.params, XT_full)).reshape(nt_eval, nx_eval, 2)
    H_pred = UV_pred[:, :, 0] ** 2 + UV_pred[:, :, 1] ** 2

    u0, v0 = uv_initial_condition_mode(x_plot)
    U_true, V_true = solve_schrodinger_ref(x_plot, t_plot, kappa_test, u0, v0)
    H_true = U_true ** 2 + V_true ** 2

    errors = [np.linalg.norm(H_pred[i] - H_true[i]) / max(np.linalg.norm(H_true[i]), 1e-10) for i in range(nt_eval)]
    norm_g = max(float(np.mean(H_true ** 2)), 1e-10)
    mse = float(np.mean((H_pred - H_true) ** 2) / norm_g)
    return {"H_pred": H_pred, "H_true": H_true, "errors": errors, "avg": float(np.mean(errors)), "mse": mse}

"""Main"""
if __name__ == "__main__":
    cfg = SchrodingerConfig()
    np.random.seed(cfg.seed)
    test_kappas = make_test_kappas(cfg)
    # Plot the held-out kappa where Padé+PINN did best so the two models' figures are directly comparable
    plot_kappa = best_pade_pinn_kappa("models/pade_pinn_schrodinger_error_metrics.json", test_kappas)

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
    print(f"  PINN training  -  single global network  (Schrodinger equation)")
    print(f"  Epochs={epochs}  lr={lr}")
    print(fr"  kappa in [{param_lb, param_ub}]  log-uniform   plot kappa={plot_kappa:.5f} (Padé+PINN best)")
    print(f"{'='*62}")

    for epoch in range(1, epochs + 1):
        X_res, X_bc_left, X_bc_right, X_ic, UV_ic = prepare_data(nx, nbc, nic, duration, min_x, max_x, starting_point)
        kappa = float(sample_log_kappa(param_lb, param_ub))
        kappa_j = jnp.asarray(kappa)

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            def residual_fn(X: jnp.ndarray) -> jnp.ndarray:
                kappa_col = jnp.full((X.shape[0], 1), kappa_j)
                return schrodinger_eqn(model.params, jnp.concatenate([X, kappa_col], axis=1), kappa_j)
            X_res_anchor = select_rar_points(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        start = time.time()
        model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc_left, X_bc_right, X_ic, UV_ic, kappa_j, current_causal_eps)
        model.params = model.get_params(model.opt_state)
        end = time.time()

        if epoch % 500 == 0:
            loss_res, loss_bc, loss_ic = losses
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}, {float(loss_ic):.3e}]  kappa={kappa:.5f}  causal_eps={float(current_causal_eps):.3f}  {end-start:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pinn_schrodinger.pkl")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_kappas)} held-out κ values")
    print("=" * 65)
    error_metric = {}

    x_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    t_plot = np.linspace(starting_point, duration, nx_eval)
    X_plot, T_plot = np.meshgrid(x_plot, t_plot)
    XT_flat = np.column_stack([X_plot.ravel(), T_plot.ravel()])

    for trial, kappa_test in enumerate(test_kappas):
        print(f"\n  [{trial+1}/{len(test_kappas)}]  κ = {kappa_test:.5f}")

        result = evaluate_kappa(model, kappa_test, x_plot, t_plot, XT_flat)
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

    json_path = "models/pinn_schrodinger_error_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved to {json_path}")

    plot_result = evaluate_kappa(model, plot_kappa, x_plot, t_plot, XT_flat)
    print(f"\nPlotting held-out κ = {plot_kappa:.5f} (Padé+PINN's best)  —  PINN Rel_MSE there = {plot_result['mse']:.3e}")

    # Real boundary samples (same distribution/code path used during training)
    _, X_bc_left, X_bc_right, _, _ = prepare_data(1, cfg.nbc_plot, 1, duration, min_x, max_x, starting_point)

    x_plot_closed = np.append(x_plot, max_x)
    X_plot_closed, T_plot_closed = np.meshgrid(x_plot_closed, t_plot)
    XT_flat_closed = jnp.array(np.column_stack([X_plot_closed.ravel(), T_plot_closed.ravel(), np.full(X_plot_closed.size, plot_kappa)]))
    UV_pred_closed = np.array(model.forward(model.params, XT_flat_closed)).reshape(nx_eval, nx_eval + 1, 2)
    H_pred_closed = UV_pred_closed[:, :, 0] ** 2 + UV_pred_closed[:, :, 1] ** 2
    H_true_closed = np.concatenate([plot_result["H_true"], plot_result["H_true"][:, :1]], axis=1)

    plot_pinn_error_evolution(t_plot, plot_result["errors"], {'\\kappa': plot_kappa}, name="schrodingerPINN", folder="pinn")
    plot_prediction(T_plot_closed, X_plot_closed, H_pred_closed, H_true_closed, X_bc_left, X_bc_right, {'\\kappa': plot_kappa}, intervals=[0.02, duration / 2, duration - 0.01], name="schrodingerPINN", folder="pinn")
    animate_prediction(T_plot_closed, X_plot_closed, H_pred_closed, H_true_closed, {'\\kappa': plot_kappa}, name="schrodingerPINN", folder="pinn")

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
