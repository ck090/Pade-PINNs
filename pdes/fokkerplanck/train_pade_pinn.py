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
from common.sampling import sample_log_kappa, sample_residual_and_bc, select_rar_points
from common.plotting import plot_prediction, animate_prediction, plot_error_evolution, plot_loss_history
from common.pade_time_approx import compute_pade_time_32, make_time_gate
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.fokkerplanck.config import FokkerPlanckConfig, make_test_kappas
from pdes.fokkerplanck.physics import THETA, X0_INIT, S0_INIT, fokker_planck_exact, symbolic_fokker_planck

configure_jax(enable_x64=True)

"""Fokker-Planck residual p_t - THETA*(p + x*p_x) - (kappa^2/2)*p_xx for the Padé+PINN ansatz p = R(x,t,kappa) + phi(t)*NN(x,t,kappa)"""
@partial(jax.jit, static_argnums=(0, 1))
def fokker_planck_residual(phi_t: Callable, R: Callable, params: Params, X: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def p_single(xt: jnp.ndarray) -> jnp.ndarray:
        xt_k = jnp.concatenate([xt, kappa.reshape(1)])
        return R(xt[0], xt[1], kappa) + phi_t(xt[1]) * pinn(xt_k[None]).squeeze()
    grad_p, hess_p = jax.grad(p_single), jax.hessian(p_single)
    def residual_single(xt: jnp.ndarray) -> jnp.ndarray:
        g, H = grad_p(xt), hess_p(xt)
        return g[1] - THETA * (p_single(xt) + xt[0] * g[0]) - 0.5 * kappa ** 2 * H[0, 0]
    return jax.vmap(residual_single)(X)

"""Padé-PINN: ansatz p = R(x,t,kappa) + phi(t)*NN(x,t,kappa), IC satisfied by construction"""
class PadePINN:
    def __init__(self, layer: list[int], lr: float, R: Callable, nn_t_term: Callable, order: int = 5, n_chunks: int = 16, weight_floor: float = 0.0) -> None:
        self.layer, self.lr, self.order = layer, lr, order
        self.n_chunks, self.weight_floor = n_chunks, weight_floor
        self.R, self.nn_t_term = R, nn_t_term
        self.params = init_params(layer, zero_last_layer=True)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(self.params)

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, params: Params, X: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, X)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> tuple[Params, Any, tuple[jnp.ndarray, jnp.ndarray]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(params, X_res, X_bc_left, X_bc_right, kappa, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc_left, X_bc_right, kappa, causal_eps)
        return self.get_params(next_opt_state), next_opt_state, (loss_res, loss_bc)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        residuals = fokker_planck_residual(self.nn_t_term, self.R, params, X_res, kappa)
        # Fokker-Planck's residual is already a single scalar per point (unlike Schrodinger's (u,v)
        # pair), so it feeds causal_residual_loss directly against X_res's own time column
        loss_residual = causal_residual_loss(residuals, X_res[:, 1], causal_eps, self.n_chunks, self.weight_floor)

        # Dirichlet BC: the density vanishes at both ends of the domain
        phi_l, phi_r = self.nn_t_term(X_bc_left[:, 1]), self.nn_t_term(X_bc_right[:, 1])
        kappa_col = jnp.full((X_bc_left.shape[0], 1), kappa)
        pl_pred = self.R(X_bc_left[:, 0], X_bc_left[:, 1], kappa) + phi_l * self.forward(params, jnp.concatenate([X_bc_left, kappa_col], axis=1)).squeeze()
        pr_pred = self.R(X_bc_right[:, 0], X_bc_right[:, 1], kappa) + phi_r * self.forward(params, jnp.concatenate([X_bc_right, kappa_col], axis=1)).squeeze()
        loss_bc = jnp.mean(pl_pred ** 2) + jnp.mean(pr_pred ** 2)
        return loss_residual, loss_bc

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> jnp.ndarray:
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc_left, X_bc_right, kappa, causal_eps)
        return loss_res + loss_bc

    def predict(self, X: jnp.ndarray, kappa: float) -> jnp.ndarray:
        kappa_j = jnp.asarray(kappa)
        pade_out = jax.vmap(lambda xt: self.R(xt[0], xt[1], kappa_j))(X)
        kappa_col = jnp.full((X.shape[0], 1), kappa_j)
        X_k = jnp.concatenate([X, kappa_col], axis=1)
        return pade_out + self.nn_t_term(X[:, 1]) * self.forward(self.params, X_k).squeeze()

    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"params": self.params, "layer": self.layer, "lr": self.lr, "order": self.order}, f)
        print(f"Model saved → {filepath}")

    # R/nn_t_term are un-picklable sympy/jax closures, so they aren't saved above --
    # rebuild them the same way __main__ does, from the (fixed) PDE and the saved gate order.
    @staticmethod
    def load_model(filepath: str) -> "PadePINN":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        x_sym, t_sym, kappa_sym, p_initial, pde_op = symbolic_fokker_planck()
        R_expr, _ = compute_pade_time_32(p_initial, x_sym, t_sym, pde_op)
        R = sp.lambdify((x_sym, t_sym, kappa_sym), R_expr.evalf(), modules="jax", cse=True)
        order = data["order"]
        model = PadePINN(data["layer"], data["lr"], R, make_time_gate(t_sym, m=order), order=order)
        model.params = data["params"]
        print(f"Model loaded ← {filepath}")
        return model

"""Evaluate Padé and Padé+PINN against the exact OU density for one kappa, over the full (x,t) eval grid"""
def evaluate_kappa(model: PadePINN, R: Callable, kappa_test: float, x_plot: np.ndarray, t_plot: np.ndarray, XT_flat: jnp.ndarray) -> Dict[str, Any]:
    nx_eval, nt_eval = len(x_plot), len(t_plot)
    kappa_j = jnp.asarray(kappa_test)

    pade_p = np.array(jax.vmap(lambda xt: R(xt[0], xt[1], kappa_j))(XT_flat)).reshape(nt_eval, nx_eval)
    U_pred = np.array(model.predict(XT_flat, kappa_test)).reshape(nt_eval, nx_eval)

    X_grid, T_grid = np.meshgrid(x_plot, t_plot)
    U_true = fokker_planck_exact(X_grid, T_grid, kappa_test)

    errors_nn = [np.linalg.norm(U_pred[i] - U_true[i]) / max(np.linalg.norm(U_true[i]), 1e-10) for i in range(nt_eval)]
    errors_pade = [np.linalg.norm(pade_p[i] - U_true[i]) / max(np.linalg.norm(U_true[i]), 1e-10) for i in range(nt_eval)]

    norm_g = max(float(np.mean(U_true ** 2)), 1e-10)
    mse_nn = float(np.mean((U_pred - U_true) ** 2) / norm_g)
    mse_pade = float(np.mean((pade_p - U_true) ** 2) / norm_g)

    mass_nn = np.trapezoid(U_pred, x_plot, axis=1)
    mass_pade = np.trapezoid(pade_p, x_plot, axis=1)

    return {
        "U_pred": U_pred, "U_true": U_true,
        "errors_nn": errors_nn, "errors_pade": errors_pade,
        "avg_nn": float(np.mean(errors_nn)), "avg_pade": float(np.mean(errors_pade)),
        "mse_nn": mse_nn, "mse_pade": mse_pade,
        "mass_err_nn": float(np.max(np.abs(mass_nn - 1.0))), "mass_err_pade": float(np.max(np.abs(mass_pade - 1.0))),
        "min_p_nn": float(U_pred.min()), "min_p_pade": float(pade_p.min()),
    }

if __name__ == "__main__":
    cfg = FokkerPlanckConfig()
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

    x_sym, t_sym, kappa_sym, p_initial, pde_op = symbolic_fokker_planck()

    print("Computing [3/2] temporal Padé approximant for the Fokker-Planck equation...")
    R_expr, taylor_coeffs = compute_pade_time_32(p_initial, x_sym, t_sym, pde_op)
    print()

    R = sp.lambdify((x_sym, t_sym, kappa_sym), R_expr.evalf(), modules="jax", cse=True)
    nn_t_term = make_time_gate(t_sym, m=gate_m)
    model = PadePINN(layers, lr, R, nn_t_term, order=gate_m, n_chunks=n_chunks, weight_floor=CAUSAL_WEIGHT_FLOOR)

    loss_history: list[tuple[int, float, float]] = []

    print(f"{'='*62}")
    print(f"  Padé-PINN training -- (Fokker-Planck)")
    print(f"  epochs={epochs}  lr={lr}")
    print(f"  time={duration} nx={nx}  min_x={min_x} max_x={max_x}")
    print(f"  theta={THETA}  x0={X0_INIT}  s0={S0_INIT}")
    print(fr"  $\kappa$ in [{param_lb, param_ub}]  log-uniform")
    print(f"{'='*62}")
    for epoch in range(1, epochs + 1):
        X_res, X_bc_left, X_bc_right = sample_residual_and_bc(nx, nbc, duration, min_x, max_x, starting_point)
        kappa = jnp.asarray(sample_log_kappa(param_lb, param_ub))

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            residual_fn = lambda X: fokker_planck_residual(model.nn_t_term, model.R, model.params, X, kappa)
            X_res_anchor = select_rar_points(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        t0 = time.time()
        model.params, model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc_left, X_bc_right, kappa, current_causal_eps)

        if epoch % 500 == 0:
            loss_res, loss_bc = losses
            loss_history.append((epoch, float(loss_res), float(loss_bc)))
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}]  kappa={float(kappa):.5f}  causal_eps={float(current_causal_eps):.3f}  {time.time() - t0:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pade_pinn_fokkerplanck.pkl")

    loss_epochs = [e for e, _, _ in loss_history]
    loss_res_hist = [v for _, v, _ in loss_history]
    loss_bc_hist = [v for _, _, v in loss_history]
    plot_loss_history(loss_epochs, loss_res_hist, loss_bc_hist, name="fokkerplanckPadePINN", folder="padepinn")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_kappas)} held-out κ values")
    print("=" * 65)

    # Dirichlet (non-periodic) domain, so both endpoints are part of the evaluation grid
    x_plot = np.linspace(min_x, max_x, nx_eval)
    t_plot = np.linspace(starting_point, duration, nx_eval)
    X_plot, T_plot = np.meshgrid(x_plot, t_plot)
    XT_flat = jnp.array(np.column_stack([X_plot.ravel(), T_plot.ravel()]))

    error_metric: Dict[float, Dict] = {}
    best_kappa, best_mse_nn = float(test_kappas[0]), float("inf")
    for trial, kappa_test in enumerate(test_kappas):
        print(f"\n  [{trial+1}/{len(test_kappas)}]  κ = {kappa_test:.5f}")

        result = evaluate_kappa(model, R, kappa_test, x_plot, t_plot, XT_flat)
        avg_nn, avg_pade = result["avg_nn"], result["avg_pade"]
        mse_nn, mse_pade = result["mse_nn"], result["mse_pade"]
        err_nn_T, err_pade_T = result["errors_nn"][-1], result["errors_pade"][-1]

        print(f"  {'='*60}")
        print(f"  Rel-L2 @ t=T   Padé={err_pade_T:.3e}   Padé+PINN={err_nn_T:.3e}")
        print(f"  {'='*60}")
        print(f"  {'Metric':<12} {'[Padé]':>12} {'[Padé+PINN]':>15}")
        print(f" {'='*60}")
        print(f"  {'Avg L2':<12} {avg_pade:>12.3e} {avg_nn:>15.3e}")
        print(f"  {'Rel_MSE':<12} {mse_pade:>12.3e} {mse_nn:>15.3e}")
        print(f"  {'Rel_RMSE':<12} {np.sqrt(mse_pade):>12.3e} {np.sqrt(mse_nn):>15.3e}")
        print(f"  {'Mass err':<12} {result['mass_err_pade']:>12.3e} {result['mass_err_nn']:>15.3e}")
        print(f"  {'Min p':<12} {result['min_p_pade']:>12.3e} {result['min_p_nn']:>15.3e}")
        print(f"  {'='*60}")

        error_metric[kappa_test] = {
            "L2": (avg_pade, avg_nn),
            "Rel_MSE": (mse_pade, mse_nn),
            "Rel_RMSE": (float(np.sqrt(mse_pade)), float(np.sqrt(mse_nn))),
            "Mass_err": (result["mass_err_pade"], result["mass_err_nn"]),
            "Min_p": (result["min_p_pade"], result["min_p_nn"]),
        }

        if mse_nn < best_mse_nn:
            best_mse_nn, best_kappa = mse_nn, float(kappa_test)

    json_path = "models/pade_pinn_fokkerplanck_error_metrics.json"
    with open(json_path, "w") as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved → {json_path}")

    # Plot only the held-out kappa where Padé+PINN achieved the lowest Rel_MSE and re-evaluate
    print(f"\nBest held-out κ = {best_kappa:.5f}  (Padé+PINN Rel_MSE = {best_mse_nn:.3e}) — generating plots for this trial")
    best_result = evaluate_kappa(model, R, best_kappa, x_plot, t_plot, XT_flat)
    U_pred, U_true = best_result["U_pred"], best_result["U_true"]
    errors_nn, errors_pade = best_result["errors_nn"], best_result["errors_pade"]

    # Real random boundary samples (same distribution/code path used during training)
    _, X_bc_left, X_bc_right = sample_residual_and_bc(1, cfg.nbc_plot, duration, min_x, max_x, starting_point)

    plot_error_evolution(t_plot, errors_pade, errors_nn, {'\\kappa': best_kappa}, name="fokkerplanckPadePINN", folder="padepinn")
    plot_prediction(T_plot, X_plot, U_pred, U_true, X_bc_left, X_bc_right, {'\\kappa': best_kappa}, intervals=[0.1, duration / 2, duration - 0.05], name="fokkerplanckPadePINN", folder="padepinn")
    animate_prediction(T_plot, X_plot, U_pred, U_true, {'\\kappa': best_kappa}, name="fokkerplanckPadePINN", folder="padepinn")

    avg_pade_L2 = float(np.mean([v["L2"][0] for v in error_metric.values()]))
    avg_nn_L2 = float(np.mean([v["L2"][1] for v in error_metric.values()]))
    avg_pade_mse = float(np.mean([v["Rel_MSE"][0] for v in error_metric.values()]))
    avg_nn_mse = float(np.mean([v["Rel_MSE"][1] for v in error_metric.values()]))
    avg_pade_rmse = float(np.mean([v["Rel_RMSE"][0] for v in error_metric.values()]))
    avg_nn_rmse = float(np.mean([v["Rel_RMSE"][1] for v in error_metric.values()]))
    avg_pade_mass = float(np.mean([v["Mass_err"][0] for v in error_metric.values()]))
    avg_nn_mass = float(np.mean([v["Mass_err"][1] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics — across {len(error_metric)} held-out κ values")
    print("=" * 65)
    print(f"  {'Metric':<12} {'[Padé]':>12} {'[Padé+PINN]':>15}")
    print(f"  {'='*60}")
    print(f"  {'Avg L2':<12} {avg_pade_L2:>12.3e} {avg_nn_L2:>15.3e}")
    print(f"  {'Rel_MSE':<12} {avg_pade_mse:>12.3e} {avg_nn_mse:>15.3e}")
    print(f"  {'Rel_RMSE':<12} {avg_pade_rmse:>12.3e} {avg_nn_rmse:>15.3e}")
    print(f"  {'Mass err':<12} {avg_pade_mass:>12.3e} {avg_nn_mass:>15.3e}")
    print(f"  {'='*60}")
