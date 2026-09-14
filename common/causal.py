import jax
import jax.numpy as jnp

# Causal residual loss (Wang, Sankaran & Perdikaris, "Respecting causality is all you need for
# training physics-informed neural networks", 2022): sort residuals by time, split into n_chunks
# equal time-chunks, and weight chunk k by exp(-eps * cumulative loss of chunks 0..k-1) so gradient
# signal only reaches later times once earlier ones have converged. residuals/t_vals must be flat 1D
# arrays of matching length -- for a multi-component PDE (e.g. u,v), concatenate the components and
# duplicate t_vals to match before calling this (see pdes/schrodinger/train_pade_pinn.py).
def causal_residual_loss(residuals: jnp.ndarray, t_vals: jnp.ndarray, eps: jnp.ndarray, n_chunks: int, weight_floor: float = 0.0) -> jnp.ndarray:
    sort_idx = jnp.argsort(t_vals)
    r_sorted = residuals[sort_idx]
    n = (r_sorted.shape[0] // n_chunks) * n_chunks
    chunk_losses = jnp.mean(r_sorted[:n].reshape(n_chunks, -1) ** 2, axis=1)
    cum_losses = jnp.concatenate([jnp.zeros(1), jnp.cumsum(chunk_losses[:-1])])
    weights = jnp.maximum(jax.lax.stop_gradient(jnp.exp(-eps * cum_losses)), weight_floor)
    return jnp.dot(weights, chunk_losses) / n_chunks

# Anneal causal_eps 0 -> eps_max linearly over the first warmup_frac of training; early on, every
# chunk gets gradient signal (uniform training) so a global baseline forms before causality is enforced.
def causal_eps_schedule(epoch: int, epochs: int, eps_max: float, warmup_frac: float = 0.5) -> float:
    warmup_epochs = warmup_frac * epochs
    return eps_max if epoch >= warmup_epochs else eps_max * (epoch / warmup_epochs)
