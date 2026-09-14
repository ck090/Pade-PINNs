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
from common.sampling import sample_residual_and_bc, select_rar_points
from common.plotting import plot_prediction, animate_prediction, plot_loss_history, plot_pinn_error_evolution, best_pade_pinn_kappa
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.burgers.config import BurgersConfig, make_test_kappas
from pdes.burgers.physics import uv_initial_condition, solve_burgers_ref

configure_jax(enable_x64=True)

# Draw nu uniformly (not log-uniform, unlike this repo's diffusion-type PDEs)
def sample_nu(nu_lb: float, nu_ub: float) -> float:
    return float(np.random.uniform(nu_lb, nu_ub))

# Sample collocation, boundary, and initial-condition points; IC points/values are on top of the
# shared residual+BC sampler since the plain PINN (unlike Padé+PINN) enforces the IC via a loss term
def prepare_data(nx: int, nbc: int, nic: int, duration: float, min_x: float, max_x: float, starting_point: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X_res, X_bc_left, X_bc_right = sample_residual_and_bc(nx, nbc, duration, min_x, max_x, starting_point)
    x_ic = np.random.uniform(min_x, max_x, size=nic)
    X_ic = jnp.array([[x, starting_point] for x in x_ic])
    U_ic = jnp.array(uv_initial_condition(x_ic))
    return X_res, X_bc_left, X_bc_right, X_ic, U_ic

# Burgers' equation: u_t + u*u_x - nu*u_xx = 0
@jax.jit
def burgers_eqn(params: Params, X: jnp.ndarray, nu: float) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def u_fn(xt: jnp.ndarray) -> jnp.ndarray:
        return pinn(xt.reshape(1, -1)).squeeze()

    def compute_pde(xt: jnp.ndarray) -> jnp.ndarray:
        u = u_fn(xt)
        u_t = jax.grad(u_fn)(xt)[1]
        u_x = jax.grad(u_fn)(xt)[0]
        u_xx = jax.hessian(u_fn)(xt)[0, 0]
        return u_t + u * u_x - nu * u_xx
    return jax.vmap(compute_pde)(X)

# PINN: single global network conditioned on physics parameter nu; 1-component output (u)
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
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, X_ic: jnp.ndarray, U_ic: jnp.ndarray, nu: float, causal_eps: jnp.ndarray) -> Tuple[Any, Tuple[float, float, float]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss)(params, X_res, X_bc_left, X_bc_right, X_ic, U_ic, nu, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc_left, X_bc_right, X_ic, U_ic, nu, causal_eps)
        return next_opt_state, (loss_res, loss_bc, loss_ic)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, X_ic: jnp.ndarray, U_ic: jnp.ndarray, nu: float, causal_eps: jnp.ndarray) -> Tuple[float, float, float]:
        N_res = X_res.shape[0]
        N_bc = X_bc_left.shape[0]
        N_ic = X_ic.shape[0]
        def _append(X: jnp.ndarray, n: int) -> jnp.ndarray:
            return jnp.concatenate([X, jnp.full((n, 1), nu)], axis=1)
        X_res_full = _append(X_res, N_res)
        X_bcl_full = _append(X_bc_left, N_bc)
        X_bcr_full = _append(X_bc_right, N_bc)
        X_ic_full = _append(X_ic, N_ic)
        # PDE residual, causally weighted
        eq = burgers_eqn(params, X_res_full, nu)
        loss_residual = causal_residual_loss(eq, X_res[:, 1], causal_eps, self.n_chunks, self.weight_floor)
        # Periodic BC: u(left) = u(right), and derivative match
        out_left = self.forward(params, X_bcl_full).squeeze()
        out_right = self.forward(params, X_bcr_full).squeeze()
        loss_boundary = jnp.mean((out_left - out_right) ** 2)

        def u_fn(xt: jnp.ndarray) -> jnp.ndarray:
            return self.forward(params, xt.reshape(1, -1)).squeeze()
        grad_u = jax.grad(u_fn)
        u_x_left = jax.vmap(lambda xt: grad_u(xt)[0])(X_bcl_full)
        u_x_right = jax.vmap(lambda xt: grad_u(xt)[0])(X_bcr_full)
        loss_boundary += jnp.mean((u_x_left - u_x_right) ** 2)
        # Initial condition
        out_ic = self.forward(params, X_ic_full).squeeze()
        loss_ic = jnp.mean((out_ic - U_ic) ** 2)
        return loss_residual, loss_boundary, loss_ic

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, X_ic: jnp.ndarray, U_ic: jnp.ndarray, nu: float, causal_eps: jnp.ndarray) -> float:
        loss_residual, loss_bc, loss_ic = self.loss_component(params, X_res, X_bc_left, X_bc_right, X_ic, U_ic, nu, causal_eps)
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

# Evaluate the PINN against the reference solver for one nu, over the full (x,t) eval grid
def evaluate_nu(model: PINN, nu_test: float, x_plot: np.ndarray, t_plot: np.ndarray, XT_flat: np.ndarray) -> Dict[str, Any]:
    nx_eval, nt_eval = len(x_plot), len(t_plot)
    XT_full = jnp.array(np.column_stack([XT_flat, np.full(XT_flat.shape[0], nu_test)]))
    U_pred = np.array(model.forward(model.params, XT_full)).reshape(nt_eval, nx_eval)

    u0 = uv_initial_condition(x_plot)
    U_true = solve_burgers_ref(x_plot, t_plot, nu_test, u0)

    errors = [np.linalg.norm(U_pred[i] - U_true[i]) / max(np.linalg.norm(U_true[i]), 1e-10) for i in range(nt_eval)]
    norm_g = max(float(np.mean(U_true ** 2)), 1e-10)
    mse = float(np.mean((U_pred - U_true) ** 2) / norm_g)
    return {"U_pred": U_pred, "U_true": U_true, "errors": errors, "avg": float(np.mean(errors)), "mse": mse}

if __name__ == "__main__":
    cfg = BurgersConfig()
    np.random.seed(cfg.seed)
    test_nus = make_test_kappas(cfg)
    # Plot the held-out nu where Padé+PINN did best so the two models' figures are directly comparable
    plot_nu = best_pade_pinn_kappa("models/pade_pinn_burgers_error_metrics.json", test_nus)

    duration, nx, nbc, nic, nx_eval = cfg.duration, cfg.nx, cfg.nbc, cfg.nic, cfg.nx_eval
    min_x, max_x, starting_point = cfg.min_x, cfg.max_x, cfg.starting_point
    layers, lr, epochs = cfg.layers, cfg.lr, cfg.epochs
    nu_lb, nu_ub = cfg.param_lb, cfg.param_ub

    causal_eps_max, n_chunks = cfg.causal_eps_max, cfg.causal_n_chunks
    CAUSAL_WARMUP_FRAC, CAUSAL_WEIGHT_FLOOR = cfg.causal_warmup_frac, cfg.causal_weight_floor
    model = PINN(layers, lr, n_chunks=n_chunks, weight_floor=CAUSAL_WEIGHT_FLOOR)

    RAR_EVERY, RAR_POOL, RAR_N_ANCHOR = cfg.rar_every, cfg.rar_pool, cfg.rar_n_anchor
    X_res_anchor = None

    print(f"\n{'='*62}")
    print(f"  PINN training  -  single global network  (Burgers' equation)")
    print(f"  Epochs={epochs}  lr={lr}")
    print(fr"  nu in [{nu_lb:.5f}, {nu_ub:.5f}]  uniform  |  IC: u(x,0)=-sin(pi*x)  plot nu={plot_nu:.5f} (Padé+PINN best)")
    print(f"{'='*62}")

    for epoch in range(1, epochs + 1):
        X_res, X_bc_left, X_bc_right, X_ic, U_ic = prepare_data(nx, nbc, nic, duration, min_x, max_x, starting_point)
        nu = sample_nu(nu_lb, nu_ub)
        nu_j = jnp.asarray(nu)

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            def residual_fn(X: jnp.ndarray) -> jnp.ndarray:
                nu_col = jnp.full((X.shape[0], 1), nu_j)
                return burgers_eqn(model.params, jnp.concatenate([X, nu_col], axis=1), nu_j)
            X_res_anchor = select_rar_points(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        t0 = time.time()
        model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc_left, X_bc_right, X_ic, U_ic, nu_j, current_causal_eps)
        model.params = model.get_params(model.opt_state)

        if epoch % 500 == 0:
            l_pde, l_bc, l_ic = losses
            print(f"  {epoch:5d}/{epochs}  [{float(l_pde):.3e}, {float(l_bc):.3e}, {float(l_ic):.3e}]  nu={nu:.5f}  causal_eps={float(current_causal_eps):.3f}  {time.time()-t0:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pinn_burgers.pkl")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_nus)} held-out ν values")
    print("=" * 65)
    error_metric = {}

    x_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    t_plot = np.linspace(starting_point, duration, nx_eval)
    X_plot, T_plot = np.meshgrid(x_plot, t_plot)
    XT_flat = np.column_stack([X_plot.ravel(), T_plot.ravel()])

    for trial, nu_test in enumerate(test_nus):
        print(f"\n  [{trial+1}/{len(test_nus)}]  ν = {nu_test:.5f}")

        result = evaluate_nu(model, nu_test, x_plot, t_plot, XT_flat)
        avg_err, mse, err_t = result["avg"], result["mse"], result["errors"][-1]

        print(f"\n{'='*62}")
        print(f"{'Metric':<12} {'[PINN]':>12}")
        print(f"{'='*62}")
        print(f"{'Rel-L2@t=T':<12} {err_t:>12.3e}")
        print(f"{'L2':<12} {avg_err:>12.3e}")
        print(f"{'Rel_MSE':<12} {mse:>12.3e}")
        print(f"{'Rel_RMSE':<12} {np.sqrt(mse):>12.3e}")
        print(f"{'='*62}")

        error_metric[nu_test] = {
            'L2': avg_err,
            'Rel_MSE': mse,
            'Rel_RMSE': float(np.sqrt(mse)),
        }

    json_path = "models/pinn_burgers_error_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved to {json_path}")

    plot_result = evaluate_nu(model, plot_nu, x_plot, t_plot, XT_flat)
    print(f"\nPlotting held-out ν = {plot_nu:.5f} (Padé+PINN's best)  —  PINN Rel_MSE there = {plot_result['mse']:.3e}")

    # Real boundary samples (same distribution/code path used during training)
    _, X_bc_left_plot, X_bc_right_plot, _, _ = prepare_data(1, cfg.nbc_plot, 1, duration, min_x, max_x, starting_point)

    plot_pinn_error_evolution(t_plot, plot_result["errors"], {r'\nu': plot_nu}, name="burgersPINN", folder="pinn")
    plot_prediction(T_plot, X_plot, plot_result["U_pred"], plot_result["U_true"], X_bc_left_plot, X_bc_right_plot, {r'\nu': plot_nu}, intervals=[0.1, duration / 2, duration - 0.05], name="burgersPINN", folder="pinn")
    animate_prediction(T_plot, X_plot, plot_result["U_pred"], plot_result["U_true"], {r'\nu': plot_nu}, name="burgersPINN", folder="pinn")

    avg_L2 = float(np.mean([v['L2'] for v in error_metric.values()]))
    avg_mse = float(np.mean([v['Rel_MSE'] for v in error_metric.values()]))
    avg_rmse = float(np.mean([v['Rel_RMSE'] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics — across {len(error_metric)} held-out ν values")
    print("=" * 65)
    print(f"{'Metric':<12} {'[PINN]':>12}")
    print("=" * 62)
    print(f"{'Avg L2':<12} {avg_L2:>12.3e}")
    print(f"{'Rel_MSE':<12} {avg_mse:>12.3e}")
    print(f"{'Rel_RMSE':<12} {avg_rmse:>12.3e}")
    print("=" * 62)
