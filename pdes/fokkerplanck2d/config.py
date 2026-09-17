from dataclasses import dataclass, field
import numpy as np

# Hyperparameters shared by both the Padé+PINN and plain-PINN training scripts for the 2D isotropic
# Ornstein-Uhlenbeck Fokker-Planck equation. Both theta (mean-reversion rate) and kappa (noise
# amplitude) are swept -- unlike the 1D package, which fixes theta=0.5 and only sweeps kappa. The
# drift+diffusion operator is isotropic and separates completely across x, y, so the exact solution
# and the Padé baseline are both literal products of two copies of the 1D formula/approximant, one
# per axis, sharing the same (theta, kappa) -- not a new 2D derivation.
#
# theta_lb/theta_ub were narrowed from an initial [0.3, 0.8] sweep after the [3/2] Padé-in-time
# denominator (a quadratic in t, position-dependent) was found to develop a real root inside
# t in (0, 2] for theta well below 0.5 -- confirmed by directly solving the denominator's quadratic
# for its roots across a fine (theta, kappa, x) grid. [0.5, 0.65] x [0.8, 1.05] was verified pole-free
# on that same grid (0 hits) with margin to spare.
@dataclass
class FokkerPlanck2DConfig:
    seed: int = 9020

    # Domain: square [min_x, max_x]^2 in space, [starting_point, duration] in time
    duration: float = 2.0
    starting_point: float = 0.0
    min_x: float = -8.0
    max_x: float = 8.0

    # Collocation / boundary(per edge) / initial-condition point counts
    nx: int = 4000
    nbc: int = 800  # split evenly across all 4 square edges
    nic: int = 800  # only consumed by the plain PINN; Padé+PINN's IC is built into the ansatz
    nx_eval: int = 48  # per axis; a full nx_eval^2 spatial grid is evaluated at each t per held-out (theta, kappa) pair
    nt_eval: int = 41  # time points for the error-over-time curve

    # Plotting points and info
    nbc_plot: int = 300

    # Network: input is (x, y, t, theta, kappa) -> 5 features
    layers: list = field(default_factory=lambda: [5] + [64] * 5 + [1])
    lr: float = 1e-3
    epochs: int = 20_000
    gate_m: int = 0  # Padé+PINN time-gate order t^(m+1)/(m+1)!

    # theta (mean-reversion rate), log-uniform -- verified pole-free (see module docstring)
    theta_lb: float = 0.5
    theta_ub: float = 0.65
    # kappa (noise amplitude), log-uniform -- same range as the 1D package
    kappa_lb: float = 0.8
    kappa_ub: float = 1.05
    n_test_params: int = 30

    # RAR: every rar_every epochs, refresh rar_n_anchor anchor points from a rar_pool candidate pool
    rar_every: int = 10_000
    rar_pool: int = 10_000
    rar_n_anchor: int = 1000

    # Causal training (Wang et al. 2022): weight the residual loss by elapsed-time causality so later
    # times only get gradient signal once earlier ones have converged
    causal_eps_max: float = 0.2
    causal_n_chunks: int = 16
    causal_warmup_frac: float = 0.8
    causal_weight_floor: float = 0.4


# 30 held-out (theta, kappa) pairs fixed before training, drawn from an RNG stream independent of the
# global np.random state so they never overlap with the per-epoch training draws
def make_test_params(cfg: FokkerPlanck2DConfig) -> tuple[np.ndarray, np.ndarray]:
    rng_test = np.random.default_rng(cfg.seed ^ 0x3DFB)
    test_thetas = np.exp(rng_test.uniform(np.log(cfg.theta_lb), np.log(cfg.theta_ub), size=cfg.n_test_params))
    test_kappas = np.exp(rng_test.uniform(np.log(cfg.kappa_lb), np.log(cfg.kappa_ub), size=cfg.n_test_params))
    return test_thetas, test_kappas
