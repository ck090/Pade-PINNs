import os
import platform
import shutil
import jax

# Call once at the top of a training script, before any jitted function is traced.
# This is the single switch controlling working precision: nothing downstream pins a dtype, so
# arrays and network parameters follow enable_x64. Keep it True for the Pade ansatz -- its residual
# needs second derivatives of a rational function carrying pi^8-scale coefficients, and in float32
# the cancellation there drives the loss to inf within a few hundred epochs.
def configure_jax(enable_x64: bool = True) -> None:
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
    jax.config.update("jax_enable_x64", enable_x64)
    # Explicitly requesting "cuda,cpu" makes JAX treat cuda as mandatory and raise
    # instead of falling back when no GPU is present (e.g. running locally), and
    # bare auto-detection can pick up the Metal plugin on Mac, which doesn't
    # support x64. Detect CUDA without touching jax (any jax.devices()/backend
    # call here would eagerly initialize -- and lock in -- the default backend).
    if "JAX_PLATFORMS" not in os.environ:
        has_cuda = platform.system() == "Linux" and shutil.which("nvidia-smi") is not None
        os.environ["JAX_PLATFORMS"] = "cuda,cpu" if has_cuda else "cpu"
    jax.config.update("jax_platforms", os.environ["JAX_PLATFORMS"])
