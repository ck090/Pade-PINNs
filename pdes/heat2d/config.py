from dataclasses import dataclass, field
import numpy as np

# Hyperparameters shared by both the Padé+PINN and plain-PINN training scripts for the 2D heat
# equation. Values here come from the original flat heat2d.py (Padé+PINN only) that this package
# replaces; `nic` (initial-condition point count) is new -- that script's Padé+PINN ansatz has no IC
# loss term, but the plain PINN counterpart needs one, so it uses the same value as the other PDEs.
@dataclass
class Heat2DConfig:
    seed: int = 9020

    # Domain: square [min_x, max_x]^2 in space, [starting_point, duration] in time
    duration: float = 2.0
    starting_point: float = 0.0
    min_x: float = -1.0
    max_x: float = 1.0

    # Collocation / boundary(per face) / initial-condition point counts
    nx: int = 6400
    nbc: int = 1200
    nic: int = 300  # only consumed by the plain PINN; Padé+PINN's IC is built into the ansatz
    nx_eval: int = 256
    nt_eval: int = 100  # time points for the error-over-time curve (heavier per-step reference solve than 1D)
    n_fd: int = 129  # interior finite-difference grid size for the numeric reference solution
    n_mc: int = 100_000  # Halton quasi-random (x,y,t) points for the global Rel_MSE estimate

    # Plotting points and info
    nbc_plot: int = 100

    # Network: input is (x, y, t, kappa) -> 4 features
    layers: list = field(default_factory=lambda: [4] + [64] * 4 + [1])
    lr: float = 1e-3
    epochs: int = 20_000
    gate_m: int = 0  # Padé+PINN time-gate order t^(m+1)/(m+1)!

    # kappa (diffusivity), log-uniform
    param_lb: float = 1e-2
    param_ub: float = 1e-1
    n_test_kappas: int = 30

    # RAR: every rar_every epochs, refresh rar_n_anchor anchor points from a rar_pool candidate pool
    rar_every: int = 10_000
    rar_pool: int = 10_000
    rar_n_anchor: int = 1000

    # Causal training (Wang et al. 2022): weight the residual loss by elapsed-time causality
    causal_eps_max: float = 0.2
    causal_n_chunks: int = 16
    causal_warmup_frac: float = 0.8
    causal_weight_floor: float = 0.4


# 30 held-out kappas fixed before training, drawn from an RNG stream independent of the global
# np.random state so they never overlap with the per-epoch training draws
def make_test_kappas(cfg: Heat2DConfig) -> np.ndarray:
    rng_test = np.random.default_rng(cfg.seed ^ 0xBEEF)
    return np.exp(rng_test.uniform(np.log(cfg.param_lb), np.log(cfg.param_ub), size=cfg.n_test_kappas))
