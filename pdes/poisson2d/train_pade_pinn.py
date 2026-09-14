import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import pickle
from functools import partial
from typing import Any, Callable, Dict

import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from common.env import configure_jax
from common.nn import Params, init_params, mlp_forward
from common.sampling import sample_log_kappa, sample_residual_and_bc_2d, select_rar_points_2d
from common.plotting import plot_prediction, animate_prediction, plot_error_evolution, plot_loss_history
from common.pade_time_approx import compute_pade_time_33, make_time_gate
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.poisson2d.config import Poisson2DConfig, make_test_kappas
from pdes.poisson2d.physics import symbolic_poisson2d, poisson2d_exact, poisson2d_steady_exact

configure_jax(enable_x64=True)

# 2D Poisson pseudo-time relaxation residual u_t - kappa*(u_xx+u_yy) - f_src(x,y) for the Padé+PINN
# ansatz u = R(x,y,t,kappa) + phi(t)*NN(x,y,t,kappa)
@partial(jax.jit, static_argnums=(0, 1, 2))
def poisson2d_residual(phi_t: Callable, R: Callable, f_src: Callable, params: Params, X: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def u_single(xyt: jnp.ndarray) -> jnp.ndarray:
        xyt_k = jnp.concatenate([xyt, kappa.reshape(1)])
        return R(xyt[0], xyt[1], xyt[2], kappa) + phi_t(xyt[2]) * pinn(xyt_k[None]).squeeze()
    grad_u, hess_u = jax.grad(u_single), jax.hessian(u_single)
    def residual_single(xyt: jnp.ndarray) -> jnp.ndarray:
        g, H = grad_u(xyt), hess_u(xyt)
        return g[2] - kappa * (H[0, 0] + H[1, 1]) - f_src(xyt[0], xyt[1])
    return jax.vmap(residual_single)(X)

# Padé-PINN: ansatz u = R(x,y,t,kappa) + phi(t)*NN(x,y,t,kappa), IC (u=0) satisfied by construction
class PadePINN:
    def __init__(self, layer: list[int], lr: float, R: Callable, nn_t_term: Callable, f_src: Callable, order: int = 5, n_chunks: int = 16, weight_floor: float = 0.0) -> None:
        self.layer, self.lr, self.order = layer, lr, order
        self.n_chunks, self.weight_floor = n_chunks, weight_floor
        self.R, self.nn_t_term, self.f_src = R, nn_t_term, f_src
        self.params = init_params(layer, zero_last_layer=True)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(self.params)

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, params: Params, X: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, X)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> tuple[Params, Any, tuple[jnp.ndarray, jnp.ndarray]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(params, X_res, X_bc, kappa, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc, kappa, causal_eps)
        return self.get_params(next_opt_state), next_opt_state, (loss_res, loss_bc)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        residuals = poisson2d_residual(self.nn_t_term, self.R, self.f_src, params, X_res, kappa)
        loss_residual = causal_residual_loss(residuals, X_res[:, 2], causal_eps, self.n_chunks, self.weight_floor)

        phi_bc = self.nn_t_term(X_bc[:, 2])
        kappa_col = jnp.full((X_bc.shape[0], 1), kappa)
        # u_bc = 0 Dirichlet condition on all boundaries
        u_bc_pred = self.R(X_bc[:, 0], X_bc[:, 1], X_bc[:, 2], kappa) + phi_bc * self.forward(params, jnp.concatenate([X_bc, kappa_col], axis=1)).squeeze()
        loss_bc = jnp.mean(u_bc_pred ** 2)
        return loss_residual, loss_bc

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> jnp.ndarray:
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc, kappa, causal_eps)
        return loss_res + loss_bc

    def predict(self, X: jnp.ndarray, kappa: float) -> jnp.ndarray:
        kappa_j = jnp.asarray(kappa)
        pade_out = jax.vmap(lambda xyt: self.R(xyt[0], xyt[1], xyt[2], kappa_j))(X)
        kappa_col = jnp.full((X.shape[0], 1), kappa_j)
        X_k = jnp.concatenate([X, kappa_col], axis=1)
        return pade_out + self.nn_t_term(X[:, 2]) * self.forward(self.params, X_k).squeeze()

    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"params": self.params, "layer": self.layer, "lr": self.lr, "order": self.order}, f)
        print(f"Model saved → {filepath}")

    # R/nn_t_term/f_src are un-picklable sympy/jax closures, so they aren't saved above --
    # rebuild them the same way __main__ does, from the (fixed) PDE and the saved gate order.
    @staticmethod
    def load_model(filepath: str) -> "PadePINN":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        x_sym, y_sym, t_sym, kappa_sym, u_initial, f_src_expr, pde_op = symbolic_poisson2d()
        R_expr, _ = compute_pade_time_33(u_initial, x_sym, t_sym, pde_op, f_src=f_src_expr)
        R = sp.lambdify((x_sym, y_sym, t_sym, kappa_sym), R_expr.evalf(), modules="jax", cse=True)
        f_src = sp.lambdify((x_sym, y_sym), f_src_expr, modules="jax")
        order = data["order"]
        model = PadePINN(data["layer"], data["lr"], R, make_time_gate(t_sym, m=order), f_src, order=order)
        model.params = data["params"]
        print(f"Model loaded ← {filepath}")
        return model

# Evaluate Padé and Padé+PINN against the exact relaxation solution for one kappa, over the full (x,y,t) eval grid
def evaluate_kappa(model: PadePINN, R: Callable, kappa_test: float, X_grid: np.ndarray, Y_grid: np.ndarray, t_plot: np.ndarray) -> Dict[str, Any]:
    kappa_j = jnp.asarray(kappa_test)

    errors_nn, errors_pade = [], []
    U_pred_full, pade_full, U_true_full = [], [], []
    for t_val in t_plot:
        T_grid = np.full_like(X_grid, t_val)
        XYT_flat = jnp.array(np.column_stack([X_grid.ravel(), Y_grid.ravel(), T_grid.ravel()]))

        pade_u_t = np.array(jax.vmap(lambda xyt: R(xyt[0], xyt[1], xyt[2], kappa_j))(XYT_flat))
        U_pred_t = np.array(model.predict(XYT_flat, kappa_test))
        U_true_t = poisson2d_exact(X_grid, Y_grid, t_val, kappa_test).ravel()

        norm_true = max(np.linalg.norm(U_true_t), 1e-10)
        errors_nn.append(np.linalg.norm(U_pred_t - U_true_t) / norm_true)
        errors_pade.append(np.linalg.norm(pade_u_t - U_true_t) / norm_true)

        U_pred_full.append(U_pred_t)
        pade_full.append(pade_u_t)
        U_true_full.append(U_true_t)

    U_pred_full, pade_full, U_true_full = np.array(U_pred_full), np.array(pade_full), np.array(U_true_full)
    avg_nn, avg_pade = float(np.mean(errors_nn)), float(np.mean(errors_pade))

    norm_g = max(float(np.mean(U_true_full ** 2)), 1e-10)
    mse_nn = float(np.mean((U_pred_full - U_true_full) ** 2) / norm_g)
    mse_pade = float(np.mean((pade_full - U_true_full) ** 2) / norm_g)

    # Steady-state check against the exact 2D Poisson solution of -kappa*Lap(u) = sin(pi x)sin(pi y)
    u_poisson_exact = poisson2d_steady_exact(X_grid, Y_grid, kappa_test)
    steady_err_nn = float(np.linalg.norm(U_pred_full[-1] - u_poisson_exact.ravel()) / max(np.linalg.norm(u_poisson_exact), 1e-10))

    return {
        "U_pred_full": U_pred_full, "U_true_full": U_true_full, "pade_full": pade_full,
        "errors_nn": errors_nn, "errors_pade": errors_pade,
        "avg_nn": avg_nn, "avg_pade": avg_pade,
        "mse_nn": mse_nn, "mse_pade": mse_pade,
        "steady_err_nn": steady_err_nn,
    }

if __name__ == "__main__":
    cfg = Poisson2DConfig()
    np.random.seed(cfg.seed)
    test_kappas = make_test_kappas(cfg)

    duration, nx, nbc, nx_eval = cfg.duration, cfg.nx, cfg.nbc, cfg.nx_eval
    min_x, max_x, starting_point = cfg.min_x, cfg.max_x, cfg.starting_point
    gate_m = cfg.gate_m
    layers, lr, epochs = cfg.layers, cfg.lr, cfg.epochs
    param_lb, param_ub = cfg.param_lb, cfg.param_ub

    RAR_EVERY, RAR_POOL, RAR_N_ANCHOR = cfg.rar_every, cfg.rar_pool, cfg.rar_n_anchor
    X_res_anchor = None

    causal_eps_max, n_chunks = cfg.causal_eps_max, cfg.causal_n_chunks
    CAUSAL_WARMUP_FRAC, CAUSAL_WEIGHT_FLOOR = cfg.causal_warmup_frac, cfg.causal_weight_floor

    x_sym, y_sym, t_sym, kappa_sym, u_initial, f_src_expr, pde_op = symbolic_poisson2d()

    print("Computing [3/3] temporal Padé approximant for 2D Poisson pseudo-time relaxation...")
    R_expr, taylor_coeffs = compute_pade_time_33(u_initial, x_sym, t_sym, pde_op, f_src=f_src_expr)
    print()

    R = sp.lambdify((x_sym, y_sym, t_sym, kappa_sym), R_expr.evalf(), modules="jax", cse=True)
    f_src = sp.lambdify((x_sym, y_sym), f_src_expr, modules="jax")
    nn_t_term = make_time_gate(t_sym, m=gate_m)
    model = PadePINN(layers, lr, R, nn_t_term, f_src, order=gate_m, n_chunks=n_chunks, weight_floor=CAUSAL_WEIGHT_FLOOR)

    loss_history: list[tuple[int, float, float]] = []

    print(f"\n{'='*62}")
    print(f"  Padé-PINN training -- (2D Poisson, pseudo-time relaxation)")
    print(f"  epochs={epochs}  lr={lr}")
    print(f"  time={duration} nx={nx}  min_x={min_x} max_x={max_x}")
    print(fr"  $\kappa$ in [{param_lb, param_ub}]  log-uniform")
    print(f"{'='*62}")
    for epoch in range(1, epochs + 1):
        X_res, X_bc = sample_residual_and_bc_2d(nx, nbc, duration, min_x, max_x, starting_point)
        kappa = jnp.asarray(sample_log_kappa(param_lb, param_ub))

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            residual_fn = lambda X: poisson2d_residual(model.nn_t_term, model.R, model.f_src, model.params, X, kappa)
            X_res_anchor = select_rar_points_2d(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        t0 = time.time()
        model.params, model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc, kappa, current_causal_eps)

        if epoch % 500 == 0:
            loss_res, loss_bc = losses
            loss_history.append((epoch, float(loss_res), float(loss_bc)))
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}]  kappa={float(kappa):.5f}  causal_eps={float(current_causal_eps):.3f}  {time.time() - t0:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pade_pinn_poisson2d.pkl")

    loss_epochs = [e for e, _, _ in loss_history]
    loss_res_hist = [v for _, v, _ in loss_history]
    loss_bc_hist = [v for _, _, v in loss_history]
    plot_loss_history(loss_epochs, loss_res_hist, loss_bc_hist, name="poisson2dPadePINN", folder="padepinn")

    print("\n" + "=" * 65)
    print(f"Evaluation - {len(test_kappas)} held-out kappa values")
    print("=" * 65)

    x_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    y_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    # Excludes t=0, where ||u_true||=0 exactly (relaxation starts from rest) — undefined for a relative metric
    t_plot = np.linspace(starting_point, duration, cfg.nt_eval)[1:]
    X_grid, Y_grid = np.meshgrid(x_plot, y_plot)

    error_metric: Dict[float, Dict] = {}
    best_kappa, best_mse_nn = float(test_kappas[0]), float("inf")
    for trial, kappa_test in enumerate(test_kappas):
        print(f"\n  [{trial+1}/{len(test_kappas)}]  kappa = {kappa_test:.5f}")

        result = evaluate_kappa(model, R, kappa_test, X_grid, Y_grid, t_plot)
        avg_nn, avg_pade = result["avg_nn"], result["avg_pade"]
        mse_nn, mse_pade = result["mse_nn"], result["mse_pade"]
        err_nn_T, err_pade_T = result["errors_nn"][-1], result["errors_pade"][-1]
        steady_err_nn = result["steady_err_nn"]

        print(f"  {'='*60}")
        print(f"  Rel-L2 @ t=T   Pade={err_pade_T:.3e}   Pade+PINN={err_nn_T:.3e}")
        print(f"  Poisson steady-state Rel-L2 (Pade+PINN) = {steady_err_nn:.3e}")
        print(f"  {'='*60}")
        print(f"  {'Metric':<12} {'[Pade]':>12} {'[Pade+PINN]':>15}")
        print(f"  {'='*60}")
        print(f"  {'Avg L2':<12} {avg_pade:>12.3e} {avg_nn:>15.3e}")
        print(f"  {'Rel_MSE':<12} {mse_pade:>12.3e} {mse_nn:>15.3e}")
        print(f"  {'Rel_RMSE':<12} {np.sqrt(mse_pade):>12.3e} {np.sqrt(mse_nn):>15.3e}")
        print(f"  {'='*60}")

        error_metric[kappa_test] = {
            "L2": (avg_pade, avg_nn),
            "Rel_MSE": (mse_pade, mse_nn),
            "Rel_RMSE": (float(np.sqrt(mse_pade)), float(np.sqrt(mse_nn))),
            "steady_state_L2_nn": steady_err_nn,
        }

        if mse_nn < best_mse_nn:
            best_mse_nn, best_kappa = mse_nn, float(kappa_test)

    json_path = "models/pade_pinn_poisson2d_error_metrics.json"
    with open(json_path, "w") as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved -> {json_path}")

    # Plot only the held-out kappa where Padé+PINN achieved the lowest Rel_MSE and re-evaluate
    print(f"\nBest held-out kappa = {best_kappa:.5f}  (Padé+PINN Rel_MSE = {best_mse_nn:.3e}) - generating plots for this trial")
    best_result = evaluate_kappa(model, R, best_kappa, X_grid, Y_grid, t_plot)

    # For visualization, we take a 1D spatial slice at y = 0.5 to reuse 1D plotting tools
    idx_y = np.argmin(np.abs(y_plot - 0.5))
    U_pred_slice = best_result["U_pred_full"][:, idx_y*nx_eval : (idx_y+1)*nx_eval]
    U_true_slice = best_result["U_true_full"][:, idx_y*nx_eval : (idx_y+1)*nx_eval]
    X_plot, T_plot = np.meshgrid(x_plot, t_plot)

    # Real random boundary samples (same distribution/code path used during training), left/right faces only
    _, X_bc_sample = sample_residual_and_bc_2d(1, cfg.nbc_plot, duration, min_x, max_x, starting_point)
    X_bc_sample = np.array(X_bc_sample)
    X_bc_left = X_bc_sample[:cfg.nbc_plot, [0, 2]]
    X_bc_right = X_bc_sample[cfg.nbc_plot:2 * cfg.nbc_plot, [0, 2]]

    plot_error_evolution(t_plot, best_result["errors_pade"], best_result["errors_nn"], {'\\kappa': best_kappa}, name="poisson2dPadePINN", folder="padepinn")
    plot_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, X_bc_left, X_bc_right, {'\\kappa': best_kappa}, intervals=[0.2, duration / 2, duration - 0.2], name="poisson2dPadePINN", folder="padepinn")
    animate_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, {'\\kappa': best_kappa}, name="poisson2dPadePINN", folder="padepinn")

    avg_pade_L2 = float(np.mean([v["L2"][0] for v in error_metric.values()]))
    avg_nn_L2 = float(np.mean([v["L2"][1] for v in error_metric.values()]))
    avg_pade_mse = float(np.mean([v["Rel_MSE"][0] for v in error_metric.values()]))
    avg_nn_mse = float(np.mean([v["Rel_MSE"][1] for v in error_metric.values()]))
    avg_pade_rmse = float(np.mean([v["Rel_RMSE"][0] for v in error_metric.values()]))
    avg_nn_rmse = float(np.mean([v["Rel_RMSE"][1] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics - across {len(error_metric)} held-out kappa values")
    print("=" * 65)
    print(f"  {'Metric':<12} {'[Pade]':>12} {'[Pade+PINN]':>15}")
    print(f"  {'='*60}")
    print(f"  {'Avg L2':<12} {avg_pade_L2:>12.3e} {avg_nn_L2:>15.3e}")
    print(f"  {'Rel_MSE':<12} {avg_pade_mse:>12.3e} {avg_nn_mse:>15.3e}")
    print(f"  {'Rel_RMSE':<12} {avg_pade_rmse:>12.3e} {avg_nn_rmse:>15.3e}")
    print(f"  {'='*60}")
