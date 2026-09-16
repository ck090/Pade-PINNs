from dataclasses import dataclass, field
import numpy as np

# Hyperparameters shared by both the Padé+PINN and plain-PINN training scripts for the 1D Allen-Cahn
# equation. Values here come from the original flat allencahn.py / pinn_allencahn.py that this package
# replaces. The flat Padé+PINN script already had causal training genuinely tuned in
# (causal_eps_max=1.0, not the 0.2 used elsewhere in this repo) -- preserved as-is. The flat plain-PINN
# script had no causal training at all; added per the standing decision to keep it on for every model,
# using the same eps_max=1.0 for consistency within this PDE.
@dataclass
class AllenCahnConfig:
    seed: int = 9020

    # Domain
    duration: float = 2.0
    starting_point: float = 0.0
    min_x: float = -1.0
    max_x: float = 1.0

    # Collocation / boundary / initial-condition point counts
    nx: int = 6400
    nbc: int = 1200
    nic: int = 200  # only consumed by the plain PINN; Padé+PINN's IC is built into the ansatz
    nx_eval: int = 256

    # Plotting points and info
    nbc_plot: int = 100

    # Network
    layers: list = field(default_factory=lambda: [3] + [64] * 4 + [1])
    lr: float = 1e-3
    epochs: int = 50_000
    gate_m: int = 0  # Padé+PINN time-gate order t^(m+1)/(m+1)!

    # kappa (diffusivity), log-uniform
    param_lb: float = 0.01
    param_ub: float = 0.1
    n_test_kappas: int = 30

    # RAR: every rar_every epochs, refresh rar_n_anchor anchor points from a rar_pool candidate pool
    rar_every: int = 10_000
    rar_pool: int = 10_000
    rar_n_anchor: int = 1000

    # Causal training (Wang et al. 2022): weight the residual loss by elapsed-time causality
    causal_eps_max: float = 1.0
    causal_n_chunks: int = 16
    causal_warmup_frac: float = 0.8
    causal_weight_floor: float = 0.4


# 30 held-out kappas fixed before training, drawn from an RNG stream independent of the global
# np.random state so they never overlap with the per-epoch training draws
def make_test_kappas(cfg: AllenCahnConfig) -> np.ndarray:
    rng_test = np.random.default_rng(cfg.seed ^ 0xBEEF)
    return np.exp(rng_test.uniform(np.log(cfg.param_lb), np.log(cfg.param_ub), size=cfg.n_test_kappas))
