from dataclasses import dataclass, field
import numpy as np

# Hyperparameters shared by both the Padé+PINN and plain-PINN training scripts for the 2D Poisson
# equation, solved via pseudo-time relaxation. Values here come from the original flat poisson2d.py /
# pinn_poisson2d.py that this package replaces. The original Padé+PINN script had causal weighting
# present but effectively disabled (causal_eps_max=0.0, causal_n_chunks=1); per the standing decision
# to keep causal training on for every model going forward, this config turns it on with the same
# defaults used for heat2d (the other 2D PDE).
@dataclass
class Poisson2DConfig:
    seed: int = 9020

    # Domain: square [min_x, max_x]^2 in space, [starting_point, duration] in pseudo-time
    duration: float = 2.0
    starting_point: float = 0.0
    min_x: float = -1.0
    max_x: float = 1.0

    # Collocation / boundary(per face) / initial-condition point counts
    nx: int = 6400
    nbc: int = 1200
    nic: int = 300  # only consumed by the plain PINN; Padé+PINN's IC (u=0) is built into the ansatz
    nx_eval: int = 256
    nt_eval: int = 51  # pseudo-time points for the error-over-time curve; t=0 is excluded at eval time (||u_true||=0 there)

    # Plotting points and info
    nbc_plot: int = 100

    # Network: input is (x, y, t, kappa) -> 4 features
    layers: list = field(default_factory=lambda: [4] + [64] * 5 + [1])
    lr: float = 1e-3
    epochs: int = 20_000
    gate_m: int = 0  # Padé+PINN time-gate order t^(m+1)/(m+1)!

    # kappa (diffusivity of the pseudo-time relaxation), log-uniform
    param_lb: float = 1e-2
    param_ub: float = 0.5
    n_test_kappas: int = 30

    # RAR: every rar_every epochs, refresh rar_n_anchor anchor points from a rar_pool candidate pool
    rar_every: int = 10_000
    rar_pool: int = 10_000
    rar_n_anchor: int = 1000

    # Causal training (Wang et al. 2022): weight the residual loss by elapsed pseudo-time causality
    causal_eps_max: float = 0.2
    causal_n_chunks: int = 16
    causal_warmup_frac: float = 0.8
    causal_weight_floor: float = 0.4


# 30 held-out kappas fixed before training, drawn from an RNG stream independent of the global
# np.random state so they never overlap with the per-epoch training draws
def make_test_kappas(cfg: Poisson2DConfig) -> np.ndarray:
    rng_test = np.random.default_rng(cfg.seed ^ 0xBEEF)
    return np.exp(rng_test.uniform(np.log(cfg.param_lb), np.log(cfg.param_ub), size=cfg.n_test_kappas))
