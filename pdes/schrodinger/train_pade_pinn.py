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
from common.pade_time_approx import compute_pade_time_22_pair
from pdes.schrodinger.config import SchrodingerConfig, make_test_kappas
from pdes.schrodinger.physics import uv_initial_condition_mode, solve_schrodinger_ref, symbolic_linearized_schrodinger

configure_jax(enable_x64=True)

# t^(m+1)/(m+1)! gate: zero at t=0 (IC enforced); first m time derivatives of ansatz determined by Padé alone
def make_time_gate(t_sym: sp.Symbol, m: int = 0) -> Callable:
    expr = t_sym ** (m + 1) / sp.factorial(m + 1)
    return sp.lambdify(t_sym, expr, modules="jax")

# Nonlinear Schrodinger residual for the Padé+PINN ansatz w = (R_re,R_im) + phi(t)*NN(x,t,kappa)
@partial(jax.jit, static_argnums=(0, 1, 2))
def schrodinger_residual(phi_t: Callable, R_re: Callable, R_im: Callable, params: Params, X: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    pinn = partial(mlp_forward, params)
    def u_single(xt: jnp.ndarray) -> jnp.ndarray:
        xt_k = jnp.concatenate([xt, kappa.reshape(1)])
        return R_re(xt[0], xt[1], kappa) + phi_t(xt[1]) * pinn(xt_k[None])[:, 0].squeeze()
    def v_single(xt: jnp.ndarray) -> jnp.ndarray:
        xt_k = jnp.concatenate([xt, kappa.reshape(1)])
        return R_im(xt[0], xt[1], kappa) + phi_t(xt[1]) * pinn(xt_k[None])[:, 1].squeeze()
    grad_u, hess_u = jax.grad(u_single), jax.hessian(u_single)
    grad_v, hess_v = jax.grad(v_single), jax.hessian(v_single)
    def residual_single(xt: jnp.ndarray) -> jnp.ndarray:
        u_val, v_val = u_single(xt), v_single(xt)
        gu, Hu = grad_u(xt), hess_u(xt)
        gv, Hv = grad_v(xt), hess_v(xt)
        h2 = u_val ** 2 + v_val ** 2
        res_u = gu[1] + kappa * Hv[0, 0] + h2 * v_val
        res_v = gv[1] - kappa * Hu[0, 0] - h2 * u_val
        return jnp.array([res_u, res_v])
    return jax.vmap(residual_single)(X)

# Padé-PINN: ansatz (u,v) = (R_re,R_im)(x,t,kappa) + phi(t)*NN(x,t,kappa), IC satisfied by construction
class PadePINN:
    def __init__(self, layer: list[int], lr: float, R_re: Callable, R_im: Callable, nn_t_term: Callable, order: int = 5) -> None:
        self.layer, self.lr, self.order = layer, lr, order
        self.R_re, self.R_im, self.nn_t_term = R_re, R_im, nn_t_term
        self.params = init_params(layer, zero_last_layer=True)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(self.params)

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, params: Params, X: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, X)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, epoch: int, opt_state: Any, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, kappa: jnp.ndarray) -> tuple[Params, Any, tuple[jnp.ndarray, jnp.ndarray]]:
        params = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(params, X_res, X_bc_left, X_bc_right, kappa)
        next_opt_state = self.opt_update(epoch, grads, opt_state)
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc_left, X_bc_right, kappa)
        return self.get_params(next_opt_state), next_opt_state, (loss_res, loss_bc)

    @partial(jax.jit, static_argnums=(0,))
    def loss_component(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, kappa: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        residuals = schrodinger_residual(self.nn_t_term, self.R_re, self.R_im, params, X_res, kappa)
        loss_residual = jnp.mean(residuals ** 2)

        n_bc = X_bc_left.shape[0]
        phi_l, phi_r = self.nn_t_term(X_bc_left[:, 1]), self.nn_t_term(X_bc_right[:, 1])
        kappa_col = jnp.full((n_bc, 1), kappa)
        out_l = self.forward(params, jnp.concatenate([X_bc_left, kappa_col], axis=1))
        out_r = self.forward(params, jnp.concatenate([X_bc_right, kappa_col], axis=1))
        Rl_re = jax.vmap(lambda xt: self.R_re(xt[0], xt[1], kappa))(X_bc_left)
        Rl_im = jax.vmap(lambda xt: self.R_im(xt[0], xt[1], kappa))(X_bc_left)
        Rr_re = jax.vmap(lambda xt: self.R_re(xt[0], xt[1], kappa))(X_bc_right)
        Rr_im = jax.vmap(lambda xt: self.R_im(xt[0], xt[1], kappa))(X_bc_right)
        ul_pred, vl_pred = Rl_re + phi_l * out_l[:, 0], Rl_im + phi_l * out_l[:, 1]
        ur_pred, vr_pred = Rr_re + phi_r * out_r[:, 0], Rr_im + phi_r * out_r[:, 1]
        loss_bc_val = jnp.mean((ul_pred - ur_pred) ** 2 + (vl_pred - vr_pred) ** 2)
        # Derivative BC: u_x(-1, t) = u_x(1, t), same for v, for full periodic BC
        pinn = partial(mlp_forward, params)
        def u_at_xt(xt: jnp.ndarray) -> jnp.ndarray:
            xt_k = jnp.concatenate([xt, kappa.reshape(1)])
            return self.R_re(xt[0], xt[1], kappa) + self.nn_t_term(xt[1]) * pinn(xt_k[None])[:, 0].squeeze()
        def v_at_xt(xt: jnp.ndarray) -> jnp.ndarray:
            xt_k = jnp.concatenate([xt, kappa.reshape(1)])
            return self.R_im(xt[0], xt[1], kappa) + self.nn_t_term(xt[1]) * pinn(xt_k[None])[:, 1].squeeze()
        du_dx = lambda xt: jax.grad(u_at_xt)(xt)[0]
        dv_dx = lambda xt: jax.grad(v_at_xt)(xt)[0]
        loss_bc_dx = jnp.mean((jax.vmap(du_dx)(X_bc_left) - jax.vmap(du_dx)(X_bc_right)) ** 2 + (jax.vmap(dv_dx)(X_bc_left) - jax.vmap(dv_dx)(X_bc_right)) ** 2)
        return loss_residual, loss_bc_val + loss_bc_dx

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params: Params, X_res: jnp.ndarray, X_bc_left: jnp.ndarray, X_bc_right: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc_left, X_bc_right, kappa)
        return loss_res + loss_bc

    def predict(self, X: jnp.ndarray, kappa: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        kappa_j = jnp.asarray(kappa)
        Rre_out = jax.vmap(lambda xt: self.R_re(xt[0], xt[1], kappa_j))(X)
        Rim_out = jax.vmap(lambda xt: self.R_im(xt[0], xt[1], kappa_j))(X)
        kappa_col = jnp.full((X.shape[0], 1), kappa_j)
        X_k = jnp.concatenate([X, kappa_col], axis=1)
        nn_out = self.forward(self.params, X_k)
        phi = self.nn_t_term(X[:, 1])
        return Rre_out + phi * nn_out[:, 0], Rim_out + phi * nn_out[:, 1]

    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"params": self.params, "layer": self.layer, "lr": self.lr, "order": self.order}, f)
        print(f"Model saved → {filepath}")

    # R_re/R_im/nn_t_term are un-picklable sympy/jax closures, so they aren't saved above --
    # rebuild them the same way __main__ does, from the (fixed) PDE and the saved gate order.
    @staticmethod
    def load_model(filepath: str) -> "PadePINN":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        x_sym, t_sym, kappa_sym, u_initial, v_initial, pde_op_u, pde_op_v = symbolic_linearized_schrodinger()
        R_u, R_v, _, _ = compute_pade_time_22_pair(u_initial, v_initial, x_sym, t_sym, pde_op_u, pde_op_v)
        R_re = sp.lambdify((x_sym, t_sym, kappa_sym), R_u.evalf(), modules="jax")
        R_im = sp.lambdify((x_sym, t_sym, kappa_sym), R_v.evalf(), modules="jax")
        order = data.get("order", 0)
        nn_t_term = make_time_gate(t_sym, m=order)
        model = PadePINN(data["layer"], data["lr"], R_re, R_im, nn_t_term, order=order)
        model.params = data["params"]
        print(f"Model loaded ← {filepath}")
        return model

# Evaluate Padé and Padé+PINN against the reference solver for one kappa, over the full (x,t) eval grid
def evaluate_kappa(model: PadePINN, R_re: Callable, R_im: Callable, kappa_test: float, x_plot: np.ndarray, t_plot: np.ndarray, XT_flat: jnp.ndarray) -> Dict[str, Any]:
    nx_eval, nt_eval = len(x_plot), len(t_plot)
    kappa_j = jnp.asarray(kappa_test)

    pade_u = np.array(jax.vmap(lambda xt: R_re(xt[0], xt[1], kappa_j))(XT_flat)).reshape(nt_eval, nx_eval)
    pade_v = np.array(jax.vmap(lambda xt: R_im(xt[0], xt[1], kappa_j))(XT_flat)).reshape(nt_eval, nx_eval)
    pade_H = pade_u ** 2 + pade_v ** 2

    U_pred, V_pred = model.predict(XT_flat, kappa_test)
    U_pred, V_pred = np.array(U_pred).reshape(nt_eval, nx_eval), np.array(V_pred).reshape(nt_eval, nx_eval)
    H_pred = U_pred ** 2 + V_pred ** 2

    u0, v0 = uv_initial_condition_mode(x_plot)
    U_true, V_true = solve_schrodinger_ref(x_plot, t_plot, kappa_test, u0, v0)
    H_true = U_true ** 2 + V_true ** 2

    errors_nn = [np.linalg.norm(H_pred[i] - H_true[i]) / max(np.linalg.norm(H_true[i]), 1e-10) for i in range(nt_eval)]
    errors_pade = [np.linalg.norm(pade_H[i] - H_true[i]) / max(np.linalg.norm(H_true[i]), 1e-10) for i in range(nt_eval)]

    norm_g = max(float(np.mean(H_true ** 2)), 1e-10)
    mse_nn = float(np.mean((H_pred - H_true) ** 2) / norm_g)
    mse_pade = float(np.mean((pade_H - H_true) ** 2) / norm_g)

    return {
        "H_pred": H_pred, "H_true": H_true,
        "errors_nn": errors_nn, "errors_pade": errors_pade,
        "avg_nn": float(np.mean(errors_nn)), "avg_pade": float(np.mean(errors_pade)),
        "mse_nn": mse_nn, "mse_pade": mse_pade,
    }

if __name__ == "__main__":
    cfg = SchrodingerConfig()
    np.random.seed(cfg.seed)
    test_kappas = make_test_kappas(cfg)

    duration, nx, nbc, nx_eval = cfg.duration, cfg.nx, cfg.nbc, cfg.nx_eval
    min_x, max_x, starting_point = cfg.min_x, cfg.max_x, cfg.starting_point
    gate_m = cfg.gate_m
    layers, lr, epochs = cfg.layers, cfg.lr, cfg.epochs
    param_lb, param_ub = cfg.param_lb, cfg.param_ub

    RAR_EVERY, RAR_POOL, RAR_N_ANCHOR = cfg.rar_every, cfg.rar_pool, cfg.rar_n_anchor
    X_res_anchor = None

    # Padé baseline stays linear; the true |w|^2*w nonlinearity is left for the PINN correction to capture
    x_sym, t_sym, kappa_sym, u_initial, v_initial, pde_op_u, pde_op_v = symbolic_linearized_schrodinger()

    print("Computing [2/2] temporal Padé approximants for the linearized Schrodinger equation...")
    R_u, R_v, taylor_u, taylor_v = compute_pade_time_22_pair(u_initial, v_initial, x_sym, t_sym, pde_op_u, pde_op_v)
    print()

    R_re = sp.lambdify((x_sym, t_sym, kappa_sym), R_u.evalf(), modules="jax")
    R_im = sp.lambdify((x_sym, t_sym, kappa_sym), R_v.evalf(), modules="jax")
    nn_t_term = make_time_gate(t_sym, m=gate_m)
    model = PadePINN(layers, lr, R_re, R_im, nn_t_term, order=gate_m)

    loss_history: list[tuple[int, float, float]] = []

    print(f"{'='*62}")
    print(f"  Padé-PINN training -- (Schrodinger)")
    print(f"  epochs={epochs}  lr={lr}")
    print(f"  time={duration} nx={nx}  min_x={min_x} max_x={max_x}")
    print(fr"  $\kappa$ in [{param_lb, param_ub}]  log-uniform")
    print(f"{'='*62}")
    for epoch in range(1, epochs + 1):
        X_res, X_bc_left, X_bc_right = sample_residual_and_bc(nx, nbc, duration, min_x, max_x, starting_point)
        kappa = jnp.asarray(sample_log_kappa(param_lb, param_ub))

        if epoch % RAR_EVERY == 0 and epoch < epochs:
            residual_fn = lambda X: schrodinger_residual(model.nn_t_term, model.R_re, model.R_im, model.params, X, kappa)
            X_res_anchor = select_rar_points(residual_fn, RAR_POOL, RAR_N_ANCHOR, duration, min_x, max_x, starting_point)
        if X_res_anchor is not None:
            X_res = jnp.concatenate([X_res, X_res_anchor], axis=0)

        t0 = time.time()
        model.params, model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc_left, X_bc_right, kappa)

        if epoch % 500 == 0:
            loss_res, loss_bc = losses
            loss_history.append((epoch, float(loss_res), float(loss_bc)))
            print(f"  {epoch:5d}/{epochs}  [{float(loss_res):.3e}, {float(loss_bc):.3e}]  kappa={float(kappa):.5f}  {time.time() - t0:.2f}s")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/pade_pinn_schrodinger.pkl")

    loss_epochs = [e for e, _, _ in loss_history]
    loss_res_hist = [v for _, v, _ in loss_history]
    loss_bc_hist = [v for _, _, v in loss_history]
    plot_loss_history(loss_epochs, loss_res_hist, loss_bc_hist, name="schrodingerPadePINN", folder="padepinn")

    print("\n" + "=" * 65)
    print(f"Evaluation — {len(test_kappas)} held-out κ values")
    print("=" * 65)

    x_plot = np.linspace(min_x, max_x, nx_eval, endpoint=False)
    t_plot = np.linspace(starting_point, duration, nx_eval)
    X_plot, T_plot = np.meshgrid(x_plot, t_plot)
    XT_flat = jnp.array(np.column_stack([X_plot.ravel(), T_plot.ravel()]))

    error_metric: Dict[float, Dict] = {}
    best_kappa, best_mse_nn = float(test_kappas[0]), float("inf")
    for trial, kappa_test in enumerate(test_kappas):
        print(f"\n  [{trial+1}/{len(test_kappas)}]  κ = {kappa_test:.5f}")

        result = evaluate_kappa(model, R_re, R_im, kappa_test, x_plot, t_plot, XT_flat)
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
        print(f"  {'='*60}")

        error_metric[kappa_test] = {
            "L2": (avg_pade, avg_nn),
            "Rel_MSE": (mse_pade, mse_nn),
            "Rel_RMSE": (float(np.sqrt(mse_pade)), float(np.sqrt(mse_nn))),
        }

        if mse_nn < best_mse_nn:
            best_mse_nn, best_kappa = mse_nn, float(kappa_test)

    json_path = "models/pade_pinn_schrodinger_error_metrics.json"
    with open(json_path, "w") as f:
        json.dump(error_metric, f, indent=4)
    print(f"\nError metrics saved → {json_path}")

    # Plot only the held-out kappa where Padé+PINN achieved the lowest Rel_MSE and re-evaluate
    print(f"\nBest held-out κ = {best_kappa:.5f}  (Padé+PINN Rel_MSE = {best_mse_nn:.3e}) — generating plots for this trial")
    best_result = evaluate_kappa(model, R_re, R_im, best_kappa, x_plot, t_plot, XT_flat)
    H_true = best_result["H_true"]
    errors_nn, errors_pade = best_result["errors_nn"], best_result["errors_pade"]

    x_plot_closed = np.append(x_plot, max_x)
    X_plot_closed, T_plot_closed = np.meshgrid(x_plot_closed, t_plot)
    XT_flat_closed = jnp.array(np.column_stack([X_plot_closed.ravel(), T_plot_closed.ravel()]))

    U_pred_c, V_pred_c = model.predict(XT_flat_closed, best_kappa)
    U_pred_c, V_pred_c = np.array(U_pred_c).reshape(nx_eval, nx_eval + 1), np.array(V_pred_c).reshape(nx_eval, nx_eval + 1)
    H_pred_closed = U_pred_c ** 2 + V_pred_c ** 2
    # H_true is exactly periodic on this grid (solved with a wraparound FD stencil), so x=max_x repeats x=min_x
    H_true_closed = np.concatenate([H_true, H_true[:, :1]], axis=1)

    # Real random boundary samples (same distribution/code path used during training)
    _, X_bc_left, X_bc_right = sample_residual_and_bc(1, cfg.nbc_plot, duration, min_x, max_x, starting_point)

    plot_error_evolution(t_plot, errors_pade, errors_nn, {'\\kappa': best_kappa}, name="schrodingerPadePINN", folder="padepinn")
    plot_prediction(T_plot_closed, X_plot_closed, H_pred_closed, H_true_closed, X_bc_left, X_bc_right, {'\\kappa': best_kappa}, intervals=[0.02, duration / 2, duration - 0.01], name="schrodingerPadePINN", folder="padepinn")
    animate_prediction(T_plot_closed, X_plot_closed, H_pred_closed, H_true_closed, {'\\kappa': best_kappa}, name="schrodingerPadePINN", folder="padepinn")

    avg_pade_L2 = float(np.mean([v["L2"][0] for v in error_metric.values()]))
    avg_nn_L2 = float(np.mean([v["L2"][1] for v in error_metric.values()]))
    avg_pade_mse = float(np.mean([v["Rel_MSE"][0] for v in error_metric.values()]))
    avg_nn_mse = float(np.mean([v["Rel_MSE"][1] for v in error_metric.values()]))
    avg_pade_rmse = float(np.mean([v["Rel_RMSE"][0] for v in error_metric.values()]))
    avg_nn_rmse = float(np.mean([v["Rel_RMSE"][1] for v in error_metric.values()]))

    print("\n" + "=" * 65)
    print(f"Final Average Error Metrics — across {len(error_metric)} held-out κ values")
    print("=" * 65)
    print(f"  {'Metric':<12} {'[Padé]':>12} {'[Padé+PINN]':>15}")
    print(f"  {'='*60}")
    print(f"  {'Avg L2':<12} {avg_pade_L2:>12.3e} {avg_nn_L2:>15.3e}")
    print(f"  {'Rel_MSE':<12} {avg_pade_mse:>12.3e} {avg_nn_mse:>15.3e}")
    print(f"  {'Rel_RMSE':<12} {avg_pade_rmse:>12.3e} {avg_nn_rmse:>15.3e}")
    print(f"  {'='*60}")
