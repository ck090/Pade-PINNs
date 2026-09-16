# PadePINNs — Results

MSE for each PDE, averaged over that PDE's held-out parameter values. Best (lowest) MSE per row is **bolded**.

| PDE | Dimensions | Sys | Padé order | # PDE Params | Padé | Padé+PINN | PINN | IC |
| --- | :-: | :-: | :-: | :-: | ------------------: | ------------------: | ------------------: | --- |
| Fokker-Planck | 1D | Linear | [3/2] | 2 | 3.297e-04 | **1.081e-06** | 4.306e-05 | p(x,0) = 𝒩(2.0, 1.5²) |
| Heat | 1D | Linear | [3/3] | 1 | 2.420e-04 | **1.271e-05** | 2.170e-03 | u(x,0) = cos(2πx) |
| Schrödinger | 1D | Nonlinear | [2/2] | 1 | 2.399e-02 | **1.001e-02** | 1.944e-02 | u(x,0) = 0.8 cos(4πx/15), v(x,0) = 0 |
| Burgers | 1D | Nonlinear | [3/3] | 1 | 4.982e-01 | **3.605e-02** | 7.729e-02 | u(x,0) = -sin(πx) |
| Allen-Cahn | 1D | Nonlinear | [3/3] | 2 | 2.239e-01 | **1.469e-04** | 9.480e-01 | u(x,0) = cos(πx) |
| Heat2D | 2D | Linear | [3/2] | 1 | 9.048e-05 | **5.101e-05** | 3.304e-04 | u(x,y,0) = sin(πx) sin(πy) |
| Poisson2D | 2D | Linear | [3/3] | 1 | 1.476e-03 | **8.422e-05** | 5.618e-03 | u(x,y,0) = 0, f(x,y) = sin(πx) sin(πy) |
| Allen-Cahn2D | 2D | Nonlinear | [3/3] | 2 | 5.535e-01 | **5.638e-03** | 9.747e-01 | u(x,y,0) = sin(x) sin(πy) |
