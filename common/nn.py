from typing import List, Dict
import numpy as np
import jax
import jax.numpy as jnp

Params = List[Dict[str, jnp.ndarray]]

# Xavier initialisation for a single layer
def xavier_init(in_dim: int, out_dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return jnp.array(std * np.random.normal(size=(in_dim, out_dim))), jnp.zeros(out_dim)

# Initialise all layer weights via Xavier. zero_last_layer=True starts the network at f(x)=0
# (useful when the network output is added directly to a prediction, e.g. a plain PINN);
# zero_last_layer=False (default) leaves it random (e.g. a Padé+PINN correction network).
def init_params(layers: list[int], zero_last_layer: bool = False) -> Params:
    params = [{"W": W, "b": b} for W, b in (xavier_init(layers[i], layers[i + 1]) for i in range(len(layers) - 1))]
    if zero_last_layer:
        params[-1] = {"W": jnp.zeros_like(params[-1]["W"]), "b": jnp.zeros_like(params[-1]["b"])}
    return params

# Tanh MLP forward pass, linear output layer
@jax.jit
def mlp_forward(params: Params, X_in: jnp.ndarray) -> jnp.ndarray:
    X = X_in
    for layer in params[:-1]:
        X = jax.nn.tanh(X @ layer["W"] + layer["b"])
    return X @ params[-1]["W"] + params[-1]["b"]
