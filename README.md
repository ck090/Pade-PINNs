# PadePINNs — Results

Error metrics for each PDE, averaged over that PDE's random held-out parameter values (30 per PDE, except Fokker-Planck which uses 40)

| PDE | Dimensions | Sys | Padé order | | Padé | | Padé+PINN | | PINN | IC |
| --- | :-: | :-: | :-: | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: | ------------------: | --- |
| | | | | L2 | MSE | L2 | MSE | L2 | MSE | |
| Fokker-Planck | 1D | Linear | [3/2] | 1.016e-02 | 3.297e-04 | 8.656e-04 | 1.081e-06 | 6.579e-03 | 4.306e-05 | p(x,0) = 𝒩(2.0, 1.5²) |
| Heat | 1D | Linear | [3/3] | 3.245e-01 | 2.420e-04 | 7.125e-02 | 1.271e-05 | 1.986e-01 | 2.170e-03 | u(x,0) = cos(2πx) |
| Schrödinger | 1D | Nonlinear | [2/2] | 1.065e-01 | 2.399e-02 | 8.448e-02 | 1.001e-02 | 1.266e-01 | 1.944e-02 | u(x,0) = 0.8 cos(4πx/15), v(x,0) = 0 |
| Burgers | 1D | Nonlinear | [3/3] | 8.790e-01 | 4.982e-01 | 2.004e-01 | 3.605e-02 | 2.970e-01 | 7.729e-02 | u(x,0) = -sin(πx) |
| Allen-Cahn | 1D | Nonlinear | [3/3] | 4.399e-01 | 2.239e-01 | 1.120e-02 | 1.469e-04 | 9.687e-01 | 9.480e-01 | u(x,0) = cos(πx) |
| Allen-Cahn2D | 2D | Nonlinear | [3/3] | 6.967e-01 | 5.535e-01 | 7.517e-02 | 5.638e-03 | 9.743e-01 | 9.747e-01 | u(x,y,0) = sin(x) sin(πy) |
| Heat2D | 2D | Linear | [3/2] | 1.476e-02 | 9.048e-05 | 6.366e-03 | 5.101e-05 | 2.740e-02 | 3.304e-04 | u(x,y,0) = sin(πx) sin(πy) |
| Poisson2D | 2D | Linear | [3/3] | 1.135e-02 | 1.476e-03 | 4.418e-03 | 8.422e-05 | 7.740e-02 | 5.618e-03 | u(x,y,0) = 0, f(x,y) = sin(πx) sin(πy) |
