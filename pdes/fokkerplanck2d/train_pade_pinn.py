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
from common.pade_time_approx import compute_pade_time_32, make_time_gate
from common.causal import causal_residual_loss, causal_eps_schedule
from pdes.fokkerplanck2d.config import FokkerPlanck2DConfig, make_test_params
from pdes.fokkerplanck2d.physics import X0_INIT, S0_INIT, fokker_planck_exact_2d, symbolic_fokker_planck_1d

configure_jax(enable_x64=True)

# The operator separates completely across x, y, so the 2D Padé baseline is just the product of two
# calls to the same 1D [3/2] approximant (one per axis, same theta/kappa) -- built once here, reused
# by __main__ and load_model alike. [3/2] matches the already-validated 1D fokkerplanck package;
# theta/kappa's safe range is enforced in config.py, not here.
def build_R2d() -> Callable:
    xi_sym, t_sym, theta_sym, kappa_sym, p_initial, pde_op = symbolic_fokker_planck_1d()
    R_expr, _ = compute_pade_time_32(p_initial, xi_sym, t_sym, pde_op)
    R_1d = sp.lambdify((xi_sym, t_sym, theta_sym, kappa_sym), R_expr.evalf(), modules="jax", cse=True)

    def R_2d(x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, theta: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
        return R_1d(x, t, theta, kappa) * R_1d(y, t, theta, kappa)

    return R_2d

"""2D Fokker-Planck residual p_t - theta*(2p + x*p_x + y*p_y) - (kappa^2/2)*(p_xx+p_yy) for the
Padé+PINN ansatz p = R(x,y,t,theta,kappa) + phi(t)*NN(x,y,t,theta,kappa)"""
@partial(jax.jit, static_argnums=(0, 1))
def fp2d_residual(phi_t: Callable, R: Callable, params: Params, X: jnp.ndarray, theta: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def p_single(xyt: jnp.ndarray) -> jnp.ndarray:
        xyt_pk = jnp.concatenate([xyt, theta.reshape(1), kappa.reshape(1)])
        return R(xyt[0], xyt[1], xyt[2], theta, kappa) + phi_t(xyt[2]) * pinn(xyt_pk[None]).squeeze()
    grad_p, hess_p = jax.grad(p_single), jax.hessian(p_single)
    def residual_single(xyt: jnp.ndarray) -> jnp.ndarray:
        x, y = xyt[0], xyt[1]
        p_val = p_single(xyt)
        g, H = grad_p(xyt), hess_p(xyt)
        drift = theta * (2 * p_val + x * g[0] + y * g[1])
        diffusion = 0.5 * kappa ** 2 * (H[0, 0] + H[1, 1])
        return g[2] - drift - diffusion
    return jax.vmap(residual_single)(X)

"""Padé-PINN: ansatz p = R(x,y,t,theta,kappa) + phi(t)*NN(x,y,t,theta,kappa), IC satisfied by construction"""
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
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc: jnp.ndarray, theta: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> tuple[Params, Any, tuple[jnp.ndarray, jnp.ndarray]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(params, X_res, X_bc, theta, kappa, causal_eps)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc, theta, kappa, causal_eps)
        return self.get_params(next_opt_state), next_opt_state, (loss_res, loss_bc)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, theta: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        residuals = fp2d_residual(self.nn_t_term, self.R, params, X_res, theta, kappa)
        loss_residual = causal_residual_loss(residuals, X_res[:, 2], causal_eps, self.n_chunks, self.weight_floor)

        # Dirichlet BC: the density vanishes on all 4 edges of the square
        n_bc = X_bc.shape[0]
        phi_bc = self.nn_t_term(X_bc[:, 2])
        pk_col = jnp.column_stack([jnp.full(n_bc, theta), jnp.full(n_bc, kappa)])
        pade_bc = jax.vmap(lambda xyt: self.R(xyt[0], xyt[1], xyt[2], theta, kappa))(X_bc)
        p_bc = pade_bc + phi_bc * self.forward(params, jnp.concatenate([X_bc, pk_col], axis=1)).squeeze()
        loss_bc = jnp.mean(p_bc ** 2)
        return loss_residual, loss_bc

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc: jnp.ndarray, theta: jnp.ndarray, kappa: jnp.ndarray, causal_eps: jnp.ndarray) -> jnp.ndarray:
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc, theta, kappa, causal_eps)
        return loss_res + loss_bc

    def predict(self, X: jnp.ndarray, theta: float, kappa: float) -> jnp.ndarray:
        theta_j, kappa_j = jnp.asarray(theta), jnp.asarray(kappa)
        pade_out = jax.vmap(lambda xyt: self.R(xyt[0], xyt[1], xyt[2], theta_j, kappa_j))(X)
        pk_col = jnp.column_stack([jnp.full(X.shape[0], theta_j), jnp.full(X.shape[0], kappa_j)])
        X_pk = jnp.concatenate([X, pk_col], axis=1)
        return pade_out + self.nn_t_term(X[:, 2]) * self.forward(self.params, X_pk).squeeze()

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
        _, t_sym, _, _, _, _ = symbolic_fokker_planck_1d()
        R = build_R2d()
        order = data["order"]
        model = PadePINN(data["layer"], data["lr"], R, make_time_gate(t_sym, m=order), order=order)
        model.params = data["params"]
        print(f"Model loaded ← {filepath}")
        return model

"""Evaluate Padé and Padé+PINN against the exact OU density for one (theta, kappa) pair, over the full (x,y,t) eval grid"""
def evaluate_theta_kappa(model: PadePINN, R: Callable, theta_test: float, kappa_test: float, X_grid: np.ndarray, Y_grid: np.ndarray, t_plot: np.ndarray) -> Dict[str, Any]:
    theta_j, kappa_j = jnp.asarray(theta_test), jnp.asarray(kappa_test)

    shape = X_grid.shape
    x_ax, y_ax = X_grid[:, 0], Y_grid[0, :]

    errors_nn, errors_pade = [], []
    mass_nn, mass_pade = [], []
    min_p_nn, min_p_pade = [], []
    mse_nn_terms, mse_pade_terms, true_sq_terms = [], [], []
    for t_val in t_plot:
        T_grid = np.full_like(X_grid, t_val)
        XYT_flat = jnp.array(np.column_stack([X_grid.ravel(), Y_grid.ravel(), T_grid.ravel()]))

        pade_p_t = np.array(jax.vmap(lambda xyt: R(xyt[0], xyt[1], xyt[2], theta_j, kappa_j))(XYT_flat))
        U_pred_t = np.array(model.predict(XYT_flat, theta_test, kappa_test))
        U_true_t = fokker_planck_exact_2d(X_grid, Y_grid, t_val, theta_test, kappa_test).ravel()

        norm_true = max(np.linalg.norm(U_true_t), 1e-10)
        errors_nn.append(np.linalg.norm(U_pred_t - U_true_t) / norm_true)
        errors_pade.append(np.linalg.norm(pade_p_t - U_true_t) / norm_true)
        mse_nn_terms.append(np.mean((U_pred_t - U_true_t) ** 2))
        mse_pade_terms.append(np.mean((pade_p_t - U_true_t) ** 2))
        true_sq_terms.append(np.mean(U_true_t ** 2))

        # Mass over the square via nested 1D trapezoids on the reshaped grid
        mass_nn.append(np.trapz(np.trapz(U_pred_t.reshape(shape), y_ax, axis=1), x_ax, axis=0))
        mass_pade.append(np.trapz(np.trapz(pade_p_t.reshape(shape), y_ax, axis=1), x_ax, axis=0))
        min_p_nn.append(float(U_pred_t.min()))
        min_p_pade.append(float(pade_p_t.min()))

    norm_g = max(float(np.mean(true_sq_terms)), 1e-10)
    mse_nn = float(np.mean(mse_nn_terms) / norm_g)
    mse_pade = float(np.mean(mse_pade_terms) / norm_g)

    return {
        "errors_nn": errors_nn, "errors_pade": errors_pade,
        "avg_nn": float(np.mean(errors_nn)), "avg_pade": float(np.mean(errors_pade)),
        "mse_nn": mse_nn, "mse_pade": mse_pade,
        "mass_err_nn": float(np.max(np.abs(np.array(mass_nn) - 1.0))), "mass_err_pade": float(np.max(np.abs(np.array(mass_pade) - 1.0))),
        "min_p_nn": float(np.min(min_p_nn)), "min_p_pade": float(np.min(min_p_pade)),
    }

if __name__ == "__main__":
    cfg = FokkerPlanck2DConfig()
    np.random.seed(cfg.seed)
    test_thetas, test_kappas = make_test_params(cfg)

    duration, nx, nbc, nx_eval, nt_eval = cfg.duration, cfg.nx, cfg.nbc, cfg.nx_eval, cfg.nt_eval
    min_x, max_x, starting_point = cfg.min_x, cfg.max_x, cfg.starting_point
    gate_m = cfg.gate_m
    layers, lr, epochs = cfg.layers, cfg.lr, cfg.epochs
    theta_lb, theta_ub, kappa_lb, kappa_ub = cfg.theta_lb, cfg.theta_ub, cfg.kappa_lb, cfg.kappa_ub

    RAR_EVERY, RAR_POOL, RAR_N_ANCHOR = cfg.rar_every, cfg.rar_pool, cfg.rar_n_anchor
    X_res_anchor = None

    causal_eps_max, n_chunks = cfg.causal_eps_max, cfg.causal_n_chunks
    CAUSAL_WARMUP_FRAC, CAUSAL_WEIGHT_FLOOR = cfg.causal_warmup_frac, cfg.causal_weight_floor

    print("Computing [3/2] temporal Padé approximant for the 1D Ornstein-Uhlenbeck building block...")
    R = build_R2d()
    print()

    _, t_sym, _, _, _, _ = symbolic_fokker_planck_1d()
    nn_t_term = make_time_gate(t_sym, m=gate_m)
    model = PadePINN(layers, lr, R, nn_t_term, order=gate_m, n_chunks=n_chunks, weight_floor=CAUSAL_WEIGHT_FLOOR)

    loss_history: list[tuple[int, float, float]] = []

    print(f"{'='*62}")
    print(f"  Padé-PINN training -- (2D Fokker-Planck, isotropic OU)")
    print(f"  epochs={epochs}  lr={lr}")
    print(f"  time={duration} nx={nx}  min_x={min_x} max_x={max_x}")
    print(f"  x0={X0_INIT}  s0={S0_INIT}")
    print(fr"  $\theta$ in [{theta_lb, theta_ub}]  $\kappa$ in [{kappa_lb, kappa_ub}]  log-uniform")
    print(f"{'='*62}")
    for epoch in range(1, epochs + 1):
        X_res, X_bc = sample_residual_and_bc_2d(nx, nbc, duration, min_x, max_x, starting_point)
        theta = jnp.asarray(sample_log_kappa(theta_lb, theta_ub))
        kappa = jnp.asarray(sample_log_kappa(kappa_lb, kappa_ub))

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            residual_fn = lambda X: fp2d_residual(model.nn_t_term, model.R, model.params, X, theta, kappa)
            X_res_anchor = select_rar_points_2d(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        current_causal_eps = jnp.asarray(causal_eps_schedule(epoch, epochs, causal_eps_max, CAUSAL_WARMUP_FRAC))
        t0 = time.time()
        model.params, model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc, theta, kappa, current_causal_eps)

        if epoch % 500 == 0:
            loss_res, loss_bc = losses
            loss_history.append((epoch, float(loss_res), float(loss_bc)))
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}]  theta={float(theta):.4f}  kappa={float(kappa):.4f}  causal_eps={float(current_causal_eps):.3f}  {time.time() - t0:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pade_pinn_fokkerplanck2d.pkl")

    loss_epochs = [e for e, _, _ in loss_history]
    loss_res_hist = [v for _, v, _ in loss_history]
    loss_bc_hist = [v for _, _, v in loss_history]
    plot_loss_history(loss_epochs, loss_res_hist, loss_bc_hist, name="fokkerplanck2dPadePINN", folder="padepinn")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_thetas)} held-out (θ, κ) pairs")
    print("=" * 65)

    x_plot = np.linspace(min_x, max_x, nx_eval)
    y_plot = np.linspace(min_x, max_x, nx_eval)
    t_plot = np.linspace(starting_point, duration, nt_eval)
    X_grid, Y_grid = np.meshgrid(x_plot, y_plot, indexing="ij")

    error_metric: Dict[str, Dict] = {}
    best_key, best_mse_nn = None, float("inf")
    for trial, (theta_test, kappa_test) in enumerate(zip(test_thetas, test_kappas)):
        print(f"\n  [{trial+1}/{len(test_thetas)}]  θ = {theta_test:.5f}  κ = {kappa_test:.5f}")

        result = evaluate_theta_kappa(model, R, theta_test, kappa_test, X_grid, Y_grid, t_plot)
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

        key = f"{theta_test:.6f}_{kappa_test:.6f}"
        error_metric[key] = {
            "theta": float(theta_test), "kappa": float(kappa_test),
            "L2": (avg_pade, avg_nn),
            "Rel_MSE": (mse_pade, mse_nn),
            "Rel_RMSE": (float(np.sqrt(mse_pade)), float(np.sqrt(mse_nn))),
            "Mass_err": (result["mass_err_pade"], result["mass_err_nn"]),
            "Min_p": (result["min_p_pade"], result["min_p_nn"]),
        }

        if mse_nn < best_mse_nn:
            best_mse_nn, best_key = mse_nn, key

    json_path = "models/pade_pinn_fokkerplanck2d_error_metrics.json"
    with open(json_path, "w") as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved → {json_path}")

    best_theta, best_kappa = error_metric[best_key]["theta"], error_metric[best_key]["kappa"]
    print(f"\nBest held-out (θ,κ) = ({best_theta:.5f}, {best_kappa:.5f})  (Padé+PINN Rel_MSE = {best_mse_nn:.3e}) — generating plots for this trial")

    best_result = evaluate_theta_kappa(model, R, best_theta, best_kappa, X_grid, Y_grid, t_plot)

    # (x,t) slice through y=X0_INIT (the IC's peak) for the plot_prediction/animate_prediction figures,
    # which are built for a 2D (x,t) grid -- the same convention heat2d uses for its own 2D->slice plots.
    x_slice = np.linspace(min_x, max_x, 512)
    t_slice = np.linspace(starting_point, duration, 512)
    X_plot, T_plot = np.meshgrid(x_slice, t_slice)
    Y_plot = np.full_like(X_plot, X0_INIT)
    XYT_slice = jnp.array(np.column_stack([X_plot.ravel(), Y_plot.ravel(), T_plot.ravel()]))

    U_pred_slice = np.array(model.predict(XYT_slice, best_theta, best_kappa)).reshape(512, 512)
    U_true_slice = fokker_planck_exact_2d(X_plot, Y_plot, T_plot, best_theta, best_kappa)

    # Real random boundary samples (same distribution/code path used during training); columns [0, 2] = (x, t), y dropped
    _, X_bc_sample = sample_residual_and_bc_2d(1, cfg.nbc_plot, duration, min_x, max_x, starting_point)
    X_bc_sample = np.array(X_bc_sample)
    X_bc_left = X_bc_sample[:cfg.nbc_plot, [0, 2]]
    X_bc_right = X_bc_sample[cfg.nbc_plot:2 * cfg.nbc_plot, [0, 2]]

    plot_error_evolution(t_plot, best_result["errors_pade"], best_result["errors_nn"], {'\\theta': best_theta, '\\kappa': best_kappa}, name="fokkerplanck2dPadePINN", folder="padepinn")
    plot_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, X_bc_left, X_bc_right, {'\\theta': best_theta, '\\kappa': best_kappa}, intervals=[0.1, duration / 2, duration - 0.05], name="fokkerplanck2dPadePINN", folder="padepinn")
    animate_prediction(T_plot, X_plot, U_pred_slice, U_true_slice, {'\\theta': best_theta, '\\kappa': best_kappa}, name="fokkerplanck2dPadePINN", folder="padepinn")

    avg_pade_L2 = float(np.mean([v["L2"][0] for v in error_metric.values()]))
    avg_nn_L2 = float(np.mean([v["L2"][1] for v in error_metric.values()]))
    avg_pade_mse = float(np.mean([v["Rel_MSE"][0] for v in error_metric.values()]))
    avg_nn_mse = float(np.mean([v["Rel_MSE"][1] for v in error_metric.values()]))
    avg_pade_rmse = float(np.mean([v["Rel_RMSE"][0] for v in error_metric.values()]))
    avg_nn_rmse = float(np.mean([v["Rel_RMSE"][1] for v in error_metric.values()]))
    avg_pade_mass = float(np.mean([v["Mass_err"][0] for v in error_metric.values()]))
    avg_nn_mass = float(np.mean([v["Mass_err"][1] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics — across {len(error_metric)} held-out (θ, κ) pairs")
    print("=" * 65)
    print(f"  {'Metric':<12} {'[Padé]':>12} {'[Padé+PINN]':>15}")
    print(f"  {'='*60}")
    print(f"  {'Avg L2':<12} {avg_pade_L2:>12.3e} {avg_nn_L2:>15.3e}")
    print(f"  {'Rel_MSE':<12} {avg_pade_mse:>12.3e} {avg_nn_mse:>15.3e}")
    print(f"  {'Rel_RMSE':<12} {avg_pade_rmse:>12.3e} {avg_nn_rmse:>15.3e}")
    print(f"  {'Mass err':<12} {avg_pade_mass:>12.3e} {avg_nn_mass:>15.3e}")
    print(f"  {'='*60}")
