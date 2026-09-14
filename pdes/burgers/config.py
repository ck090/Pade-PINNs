from dataclasses import dataclass, field
import numpy as np

# Hyperparameters shared by both the Padé+PINN and plain-PINN training scripts for Burgers' equation.
# Values here come from the original flat burgers.py / pinn_burgers.py that this package replaces;
# both files already had causal training tuned in (n_chunks=25, unlike the n_chunks=16 default used
# elsewhere in this repo), so those values are preserved as-is rather than reset to the default.
@dataclass
class BurgersConfig:
    seed: int = 9020

    # Domain
    duration: float = 2.0
    starting_point: float = 0.0
    min_x: float = -1.0
    max_x: float = 1.0

    # Collocation / boundary / initial-condition point counts
    nx: int = 6400
    nbc: int = 1600
    nic: int = 300  # only consumed by the plain PINN; Padé+PINN's IC is built into the ansatz
    nx_eval: int = 256

    # Plotting points and info
    nbc_plot: int = 100

    # Network
    layers: list = field(default_factory=lambda: [3] + [64] * 5 + [1])
    lr: float = 1e-3
    epochs: int = 20_000
    gate_m: int = 0  # Padé+PINN time-gate order t^(m+1)/(m+1)!

    # nu (viscosity), uniform. Classic Burgers shock IC: u(x,0) = -sin(pi*x)
    param_lb: float = 0.01 / np.pi
    param_ub: float = 0.10 / np.pi
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


# 30 held-out nu values fixed before training, drawn from an RNG stream independent of the global
# np.random state so they never overlap with the per-epoch training draws
def make_test_kappas(cfg: BurgersConfig) -> np.ndarray:
    rng_test = np.random.default_rng(cfg.seed ^ 0xBEEF)
    return rng_test.uniform(cfg.param_lb, cfg.param_ub, size=cfg.n_test_kappas)
