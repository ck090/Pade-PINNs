# PadePINNs — Results

Error metrics for each PDE, averaged over that PDE's held-out κ values (30 for Schrödinger, 40 for
Fokker-Planck). All six numbers below are from causal-trained runs (Wang et al. 2022 causal residual
weighting) for both the Padé+PINN and the plain PINN.

| PDE | Dimensions | Linear / Nonlinear | Padé order | Padé | Padé+PINN | PINN | Initial Condition |
| --- | :-: | :-: | :-: | ------------------: | ------------------: | ------------------: | --- |
| Schrödinger | 1D + t | Nonlinear | [2/2] | L2: 1.065e-01<br>MSE: 2.399e-02 | L2: 8.448e-02<br>MSE: 1.001e-02 | L2: 1.266e-01<br>MSE: 1.944e-02 | u(x,0) = 0.8 cos(4πx/15), v(x,0) = 0 |
| Fokker-Planck | 1D + t | Linear | [3/2] | L2: 1.016e-02<br>MSE: 3.297e-04 | L2: 8.656e-04<br>MSE: 1.081e-06 | L2: 6.579e-03<br>MSE: 4.306e-05 | p(x,0) = 𝒩(2.0, 1.5²) |
