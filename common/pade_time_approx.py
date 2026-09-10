import sympy as sp
from typing import Callable

# t^(m+1)/(m+1)! gate for the Padé+PINN ansatz R + phi(t)*NN: zero at t=0 so the IC is exact; the first m time derivatives of the ansatz come from the Padé alone
def make_time_gate(t_sym: sp.Symbol, m: int = 0) -> Callable:
    expr = t_sym ** (m + 1) / sp.factorial(m + 1)
    return sp.lambdify(t_sym, expr, modules="jax")

def _taylor_time_coeffs(u0_expr: sp.Expr, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], order: int = 3, normalize: Callable[[sp.Expr], sp.Expr] | None = None) -> list[sp.Expr]:
    # normalize defaults to sp.simplify (readable output); pass sp.expand for PDEs with trig/exp coefficients, where simplify's identity-search is combinatorially slow but coeff() only needs expanded (not simplified) form
    # _norm = normalize if normalize is not None else sp.simplify
    f = [u0_expr]
    u_trunc = f[0]
    for n in range(order):
        N_val = sp.expand(pde_op(u_trunc))
        coeff_n = N_val.coeff(t_sym, n)
        f_next = coeff_n * sp.factorial(n)
        f.append(f_next)
        if n + 1 < order:
            u_trunc = u_trunc + f_next / sp.factorial(n + 1) * t_sym ** (n + 1)
    return f

# [2/1] Padé-in-time approximant P = (a0+a1t+a2t²)/(1+b1t) via PDE Taylor recursion
def compute_pade_time_21(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], f_src: sp.Expr = sp.Integer(0)) -> tuple[sp.Expr, list[sp.Expr]]:
    full_op = lambda u: pde_op(u) + f_src
    f0, f1, f2, f3 = _taylor_time_coeffs(u0_expr, t_sym, full_op, order=3)
    print(f" f0 = {f0}")
    print(f" f1 = {f1}")
    print(f" f2 = {f2}")
    print(f" f3 = {f3}")

    b1 = sp.simplify(-(f3 / (3 * f2)))
    print(f" b0 = 1  (normalised)")
    print(f" b1 = {b1}")

    a0 = sp.simplify(f0)
    a1 = sp.simplify(f1 + b1 * f0)
    a2 = sp.simplify(f2 / 2 + b1 * f1)
    print(f" a0 = {a0}")
    print(f" a1 = {a1}")
    print(f" a2 = {a2}")

    numer = a0 + a1 * t_sym + a2 * t_sym ** 2
    denom = 1 + b1 * t_sym
    P = numer / denom
    print(f" P[2/1](x,t) = {P}")
    return P, [f0, f1, f2, f3]

# [2/2] Padé-in-time approximant P = (a0+a1t+a2t²)/(1+b1t+b2t²) via PDE Taylor recursion
def compute_pade_time_22(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], f_src: sp.Expr = sp.Integer(0)) -> tuple[sp.Expr, list[sp.Expr]]:
    full_op = lambda u: pde_op(u) + f_src
    f0, f1, f2, f3, f4 = _taylor_time_coeffs(u0_expr, t_sym, full_op, order=4)
    print(f" f0 = {f0}")
    print(f" f1 = {f1}")
    print(f" f2 = {f2}")
    print(f" f3 = {f3}")
    print(f" f4 = {f4}")

    denom_det = 2 * f1 * f3 - 3 * f2 ** 2
    b1 = -(f1 * f4 - 2 * f2 * f3) / (2 * denom_det)
    b2 = (3 * f2 * f4 - 4 * f3 ** 2) / (12 * denom_det)
    print(f" b0 = 1  (normalised)")
    print(f" b1 = {b1}")
    print(f" b2 = {b2}")

    a0 = f0
    a1 = -(f0 * f1 * f4 - 2 * f0 * f2 * f3 - 4 * f1 ** 2 * f3 + 6 * f1 * f2 ** 2) / (2 * denom_det)
    a2 = (3 * f0 * f2 * f4 - 4 * f0 * f3 ** 2 - 6 * f1 ** 2 * f4 + 24 * f1 * f2 * f3 - 18 * f2 ** 3) / (12 * denom_det)
    print(f" a0 = {a0}")
    print(f" a1 = {a1}")
    print(f" a2 = {a2}")

    numer = a0 + a1 * t_sym + a2 * t_sym ** 2
    denom = 1 + b1 * t_sym + b2 * t_sym ** 2
    P = numer / denom
    print(f" P[2/2](x,t) = {P}")
    return P, [f0, f1, f2, f3, f4]

# Compute [2/2] Pade with non-zero forcing term in the denominator for a given PDE initial condition
def compute_pade_time_22_forced(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], f_src: sp.Expr) -> tuple[sp.Expr, list[sp.Expr]]:
    full_op = lambda u: pde_op(u) + f_src
    f0, f1, f2, f3 = _taylor_time_coeffs(u0_expr, t_sym, full_op, order=3)
    print(f" f0 = {f0}")
    print(f" f1 = {f1}")
    print(f" f2 = {f2}")
    print(f" f3 = {f3}")

    epsilon = 1e-12    
    inner_sqrt = sp.sqrt(4 * f1**2 - 2 * f0 * f2)
    denom_b1 = 6 * (2 * f0 * f2 - 3 * f1**2) + epsilon
    b1_squared = (f3 * (f1 + inner_sqrt)) / denom_b1
    b1 = sp.simplify(sp.sqrt(b1_squared))
    denom_b2 = 12 * f0 * b1 + epsilon
    b2 = sp.simplify(-(6 * f1 * b1**2 + f3) / denom_b2)
    print(f" b0 = 1  (normalised)")
    print(f" b1 = {b1}")
    print(f" b2 = {b2}")

    a0 = sp.simplify(f0)
    a1 = sp.simplify(f1)
    a2 = sp.simplify(f2 / 2 + f0 * b1 ** 2)
    print(f" a0 = {a0}")
    print(f" a1 = {a1}")
    print(f" a2 = {a2}")

    numer = a0 + a1 * t_sym + a2 * t_sym ** 2
    denom = 1 + (b1 * t_sym + b2 * t_sym ** 2) ** 2
    P = numer / denom
    print(f" P[2/2](x,t) with forcing term in denominator = {P}")
    return P, [f0, f1, f2, f3]

# [3/2] Pade-in-time approximant P = (a0+a1t+a2t²+a3t³)/(1+b1t+b2t²) via PDE Taylor recursion
def compute_pade_time_32(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], f_src: sp.Expr = sp.Integer(0)) -> tuple[sp.Expr, list[sp.Expr]]:
    full_op = lambda u: pde_op(u) + f_src
    f0, f1, f2, f3, f4, f5 = _taylor_time_coeffs(u0_expr, t_sym, full_op, order=5)
    print(f" f0 = {f0}")
    print(f" f1 = {f1}")
    print(f" f2 = {f2}")
    print(f" f3 = {f3}")
    print(f" f4 = {f4}")
    print(f" f5 = {f5}")

    Delta = sp.simplify(4 * f3**2 - 3 * f2 * f4)
    b1 = sp.simplify((3 * f2 * f5 - 5 * f3 * f4) / (5 * Delta))
    b2 = sp.simplify((5 * f4**2 - 4 * f3 * f5) / (20 * Delta))
    print(f" b0 = 1  (normalised)")
    print(f" b1 = {b1}")
    print(f" b2 = {b2}")

    a0 = sp.simplify(f0)
    a1 = sp.simplify(f1 + b1 * f0)
    a2 = sp.simplify(f2 / 2 + b1 * f1 + b2 * f0)
    a3 = sp.simplify(f3 / 6 + b1 * (f2 / 2) + b2 * f1)
    print(f" a0 = {a0}")
    print(f" a1 = {a1}")
    print(f" a2 = {a2}")
    print(f" a3 = {a3}")

    numer = a0 + a1 * t_sym + a2 * t_sym ** 2 + a3 * t_sym ** 3
    denom = 1 + b1 * t_sym + b2 * t_sym ** 2
    P = numer / denom
    print(f" P[3/2](x,t) = {P}")
    return P, [f0, f1, f2, f3, f4, f5]

# [3/3] Pade-in-time approximant P = (a0+a1t+a2t²+a3t³)/(1+b1t+b2t²+b3t³) via PDE Taylor recursion
def compute_pade_time_33(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], f_src: sp.Expr = sp.Integer(0), normalize: Callable[[sp.Expr], sp.Expr] | None = None, coeff_normalize: Callable[[sp.Expr], sp.Expr] | None = None) -> tuple[sp.Expr, list[sp.Expr]]:
    # normalize is passed through to the Taylor recursion (see _taylor_time_coeffs); coeff_normalize is applied to the final b1..b3/a0..a3 combination instead, since that step cubes/squares the f-values (e.g. c3**3) and blindly reusing an expand-based normalize there can be far more expensive than the recursion itself -- both default to sp.simplify, matching the original unparameterised behaviour exactly
    _norm = normalize if normalize is not None else sp.simplify
    _coeff_norm = coeff_normalize if coeff_normalize is not None else sp.simplify
    full_op = lambda u: pde_op(u)
    f0, f1, f2, f3, f4, f5, f6 = _taylor_time_coeffs(u0_expr, t_sym, full_op, order=6, normalize=_norm)
    print(f" f0={f0}\n f1={f1}\n f2={f2}\n f3={f3}\n f4={f4}\n f5={f5}\n f6={f6}")

    c0, c1, c2, c3, c4, c5, c6 = f0, f1, f2/2, f3/6, f4/24, f5/120, f6/720

    D  = c3**3 - 2*c2*c3*c4 + c2**2*c5 + c1*c4**2 - c1*c3*c5
    D1 = -c3**2*c4 + c2*c4**2 + c2*c3*c5 - c2**2*c6 - c1*c4*c5 + c1*c3*c6
    D2 = -c3**2*c5 + c2*c3*c6 + c3*c4**2 - c2*c4*c5 - c1*c4*c6 + c1*c5**2
    D3 = -c3**2*c6 + 2*c3*c4*c5 + c2*c4*c6 - c2*c5**2 - c4**3

    b1, b2, b3 = _coeff_norm(D1/D), _coeff_norm(D2/D), _coeff_norm(D3/D)
    print(f" b0=1 (normalised)\n b1={b1}\n b2={b2}\n b3={b3}")

    a0 = _coeff_norm(c0)
    a1 = _coeff_norm(c1 + c0*b1)
    a2 = _coeff_norm(c2 + c1*b1 + c0*b2)
    a3 = _coeff_norm(c3 + c2*b1 + c1*b2 + c0*b3)
    print(f" a0={a0}\n a1={a1}\n a2={a2}\n a3={a3}")

    numer = a0 + a1*t_sym + a2*t_sym**2 + a3*t_sym**3
    denom = 1 + b1*t_sym + b2*t_sym**2 + b3*t_sym**3
    P = numer / denom
    print(f" P[3/3](x,t) = {P}")
    return P, [f0, f1, f2, f3, f4, f5, f6]


def compute_pade_time_22_new(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], f_src: sp.Expr = sp.Integer(0), normalize: Callable[[sp.Expr], sp.Expr] | None = None, coeff_normalize: Callable[[sp.Expr], sp.Expr] | None = None) -> tuple[sp.Expr, list[sp.Expr]]:
    # normalize is passed through to the Taylor recursion (see _taylor_time_coeffs); coeff_normalize is applied to the final b1..b3/a0..a3 combination instead, since that step cubes/squares the f-values (e.g. c3**3) and blindly reusing an expand-based normalize there can be far more expensive than the recursion itself -- both default to sp.simplify, matching the original unparameterised behaviour exactly
    _norm = normalize if normalize is not None else sp.simplify
    _coeff_norm = coeff_normalize if coeff_normalize is not None else sp.simplify
    full_op = lambda u: pde_op(u)
    f0, f1, f2, f3 = _taylor_time_coeffs(u0_expr, t_sym, full_op, order=3)
    print(f" f0={f0}\n f1={f1}\n f2={f2}\n f3={f3}")

    b1 = -f3 / (3 * f2)
    print(f" b0=1 (normalised)\n b1={b1}")

    a0 = f0
    a1 = (f1 + b1 * f0)
    a2 = (f2 / 2 + b1 * f1)
    print(f" a0={a0}\n a1={a1}\n a2={a2}")

    numer = a0 + a1*t_sym + a2*t_sym**2
    denom = 1 + b1*t_sym
    P = numer / denom
    print(f" P[2/2](x,t) = {P}")
    return P, [f0, f1, f2, f3]

# Coupled Taylor-in-t recursion for a linear pair u_t = pde_op_u(v), v_t = pde_op_v(u)
def _taylor_time_coeffs_coupled(u0_expr: sp.Expr, v0_expr: sp.Expr, t_sym: sp.Symbol, pde_op_u: Callable[[sp.Expr], sp.Expr], pde_op_v: Callable[[sp.Expr], sp.Expr], order: int = 3, normalize: Callable[[sp.Expr], sp.Expr] | None = None) -> tuple[list[sp.Expr], list[sp.Expr]]:
    _fast = normalize if normalize is not None else (lambda expr: sp.cancel(sp.expand(expr)))
    f_u, f_v = [_fast(u0_expr)], [_fast(v0_expr)]
    u_trunc, v_trunc = f_u[0], f_v[0]
    for n in range(order):
        f_next_u = _fast(sp.expand(pde_op_u(v_trunc)).coeff(t_sym, n) * sp.factorial(n))
        f_next_v = _fast(sp.expand(pde_op_v(u_trunc)).coeff(t_sym, n) * sp.factorial(n))
        f_u.append(f_next_u)
        f_v.append(f_next_v)
        if n + 1 < order:
            u_trunc = _fast(u_trunc + f_next_u / sp.factorial(n + 1) * t_sym ** (n + 1))
            v_trunc = _fast(v_trunc + f_next_v / sp.factorial(n + 1) * t_sym ** (n + 1))
    return f_u, f_v

# [2/1] Padé for a linearly-coupled pair u_t = pde_op_u(v), v_t = pde_op_v(u)
def compute_pade_time_21_pair(u0_expr: sp.Expr, v0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op_u: Callable[[sp.Expr], sp.Expr], pde_op_v: Callable[[sp.Expr], sp.Expr]) -> tuple[sp.Expr, sp.Expr, list[sp.Expr], list[sp.Expr]]:
    taylor_u, taylor_v = _taylor_time_coeffs_coupled(u0_expr, v0_expr, t_sym, pde_op_u, pde_op_v, order=3)
    f0_u, f1_u, f2_u, f3_u = taylor_u
    f0_v, f1_v, f2_v, f3_v = taylor_v
    print(f" f0_u = {f0_u}   f0_v = {f0_v}")
    print(f" f1_u = {f1_u}   f1_v = {f1_v}")
    print(f" f2_u = {f2_u}   f2_v = {f2_v}")
    print(f" f3_u = {f3_u}   f3_v = {f3_v}")

    _fast = lambda expr: sp.cancel(sp.expand(expr))
    if _fast(f2_u) != 0:
        b1 = -(f3_u / (3 * f2_u))
    elif _fast(f2_v) != 0:
        b1 = -(f3_v / (3 * f2_v))
    else:
        raise ValueError("compute_pade_time_21_pair: f2_u and f2_v both vanish identically, b1 is undetermined")
    print(f" b1 (shared) = {b1}")

    def pade_21_numer(f0: sp.Expr, f1: sp.Expr, f2: sp.Expr) -> sp.Expr:
        a0 = f0
        a1 = f1 + b1 * f0
        a2 = f2 / 2 + b1 * f1
        return a0 + a1 * t_sym + a2 * t_sym ** 2

    denom = 1 + b1 * t_sym
    taylor_u.append(b1)
    P_u = pade_21_numer(f0_u, f1_u, f2_u) / denom
    P_v = pade_21_numer(f0_v, f1_v, f2_v) / denom
    print(f" P_u[2/1](x,t) = {P_u}")
    print(f" P_v[2/1](x,t) = {P_v}")
    return P_u, P_v, taylor_u, taylor_v

# [2/2] Padé for a linearly-coupled pair u_t = pde_op_u(v), v_t = pde_op_v(u)
def compute_pade_time_22_pair(u0_expr: sp.Expr, v0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op_u: Callable[[sp.Expr], sp.Expr], pde_op_v: Callable[[sp.Expr], sp.Expr]) -> tuple[sp.Expr, sp.Expr, list[sp.Expr], list[sp.Expr]]:
    taylor_u, taylor_v = _taylor_time_coeffs_coupled(u0_expr, v0_expr, t_sym, pde_op_u, pde_op_v, order=4)
    f0_u, f1_u, f2_u, f3_u, f4_u = taylor_u
    f0_v, f1_v, f2_v, f3_v, f4_v = taylor_v
    print(f" f0_u = {f0_u}   f0_v = {f0_v}")
    print(f" f1_u = {f1_u}   f1_v = {f1_v}")
    print(f" f2_u = {f2_u}   f2_v = {f2_v}")
    print(f" f3_u = {f3_u}   f3_v = {f3_v}")
    print(f" f4_u = {f4_u}   f4_v = {f4_v}")

    denom_det_u = 2 * f1_u * f3_u - 3 * f2_u ** 2
    b1_u = -(f1_u * f4_u - 2 * f2_u * f3_u) / (2 * denom_det_u)
    b2_u = (3 * f2_u * f4_u - 4 * f3_u ** 2) / (12 * denom_det_u)
    print(f" b0_u = 1  (normalised)")
    print(f" b1_u = {b1_u}")
    print(f" b2_u = {b2_u}")

    denom_det_v = 2 * f1_v * f3_v - 3 * f2_v ** 2
    b1_v = -(f1_v * f4_v - 2 * f2_v * f3_v) / (2 * denom_det_v)
    b2_v = (3 * f2_v * f4_v - 4 * f3_v ** 2) / (12 * denom_det_v)
    print(f" b0_v = 1  (normalised)")
    print(f" b1_v = {b1_v}")
    print(f" b2_v = {b2_v}")

    # denom_det is a parameter, not a closure: it was previously captured by name, so both calls
    def pade_22_numer(f0: sp.Expr, f1: sp.Expr, f2: sp.Expr, f3: sp.Expr, f4: sp.Expr, denom_det: sp.Expr) -> sp.Expr:
        a0 = f0
        a1 = -(f0 * f1 * f4 - 2 * f0 * f2 * f3 - 4 * f1 ** 2 * f3 + 6 * f1 * f2 ** 2) / (2 * denom_det)
        a2 = (3 * f0 * f2 * f4 - 4 * f0 * f3 ** 2 - 6 * f1 ** 2 * f4 + 24 * f1 * f2 * f3 - 18 * f2 ** 3) / (12 * denom_det)
        return a0, a1, a2

    a0_u, a1_u, a2_u = pade_22_numer(f0_u, f1_u, f2_u, f3_u, f4_u, denom_det_u)
    a0_v, a1_v, a2_v = pade_22_numer(f0_v, f1_v, f2_v, f3_v, f4_v, denom_det_v)
    numer_u = a0_u + a1_u * t_sym + a2_u * t_sym ** 2
    numer_v = a0_v + a1_v * t_sym + a2_v * t_sym ** 2

    P_u = numer_u / (1 + b1_u * t_sym + b2_u * t_sym ** 2)
    P_v = numer_v / (1 + b1_v * t_sym + b2_v * t_sym ** 2)
    print(f" P[2/2](x,t) = {P_u}")
    print(f" P[2/2](x,t) = {P_v}")
    return P_u, P_v, [f0_u, f1_u, f2_u, f3_u, f4_u], [f0_v, f1_v, f2_v, f3_v, f4_v]

# Degree in x of term's polynomial prefactor. Factors in which x enters non-polynomially -- the
# exp(-(x-x0)**2/(2*sigma**2)) envelope carried through the recursion -- are bounded rather than
# growing, so they contribute no x-order and are skipped; only the polynomial prefactor is measured.
def _x_degree(term: sp.Expr, x_sym: sp.Symbol) -> int:
    poly_part = sp.Integer(1)
    for factor in sp.Mul.make_args(term):
        if factor.has(x_sym) and factor.is_polynomial(x_sym):
            poly_part *= factor
    return sp.Poly(poly_part, x_sym).degree() if poly_part.has(x_sym) else 0

# Drop every additive term of expr whose x-degree exceeds max_degree
def _drop_high_x_terms(expr: sp.Expr, x_sym: sp.Symbol, max_degree: int) -> sp.Expr:
    kept = [term for term in sp.Add.make_args(sp.expand(expr)) if _x_degree(term, x_sym) <= max_degree]
    return sp.Add(*kept) if kept else sp.Integer(0)

# Truncate a rational approximant P = N(x,t)/D(x,t) to x-degree <= max_degree, applying the cut to
# the numerator and the denominator separately (P itself is a quotient, so it has no single
# expansion in x to truncate). Powers of t and of any parameter symbol ride along untouched.
# NOTE: cutting the denominator moves the approximant's poles, which is the part of a Padé that
# carries its extrapolation behaviour -- check the truncated poles before trusting it far in t.
def truncate_x_powers_expr(P: sp.Expr, x_sym: sp.Symbol, max_degree: int = 4) -> sp.Expr:
    numer, denom = sp.fraction(sp.together(P))
    numer_t = _drop_high_x_terms(numer, x_sym, max_degree)
    denom_t = _drop_high_x_terms(denom, x_sym, max_degree)
    if denom_t == 0:
        raise ValueError(f"truncate_x_powers_expr: denominator vanishes entirely at max_degree={max_degree}")
    return numer_t / denom_t

# Truncate the (P_u, P_v) pair returned by compute_pade_time_*_pair to x-degree <= max_degree
def truncate_x_powers(P_u: sp.Expr, P_v: sp.Expr, x_sym: sp.Symbol, max_degree: int = 4) -> tuple[sp.Expr, sp.Expr]:
    out = []
    for name, P in (("P_u", P_u), ("P_v", P_v)):
        numer, denom = sp.fraction(sp.together(P))
        before = [len(sp.Add.make_args(sp.expand(e))) for e in (numer, denom)]
        P_trunc = truncate_x_powers_expr(P, x_sym, max_degree)
        numer_t, denom_t = sp.fraction(P_trunc)
        after = [len(sp.Add.make_args(e)) for e in (numer_t, denom_t)]
        print(f" {name} truncated to x^{max_degree}: numer {before[0]} -> {after[0]} terms, denom {before[1]} -> {after[1]} terms")
        out.append(P_trunc)
    return out[0], out[1]

# [1/1] Padé in s=t² for symmetric 2nd-order-in-time PDEs: u_tt = pde_op(u), zero initial velocity (u_t(x,0)=0, no damping)
def compute_pade_time2_even_11(u0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr]) -> tuple[sp.Expr, list[sp.Expr]]:
    f0 = u0_expr
    print(f" f0 = {f0}")
    f2 = sp.simplify(pde_op(f0))
    print(f" f2 = {f2}")
    f4 = sp.simplify(pde_op(f2))
    print(f" f4 = {f4}")

    c0, c1, c2 = f0, sp.simplify(f2 / 2), sp.simplify(f4 / 24)
    try:
        b1 = sp.simplify(-c2 / c1)
    except Exception:
        b1 = sp.Integer(0)
    print(f"  b1 = {b1}")

    a0 = c0
    a1 = sp.simplify(c1 + c0 * b1)
    print(f"  a0 = {a0}")
    print(f"  a1 = {a1}")

    numer = a0 + a1 * t_sym ** 2
    denom = 1 + b1 * t_sym ** 2
    P = numer / denom
    print(f"  P[1/1](x,t²) = {P}")

    return P, [f0, sp.Integer(0), f2, sp.Integer(0), f4]

# [2/2] Padé for 2nd-order-in-time PDEs: u_tt = pde_op(u) - damping_op(u_t), IC u(x,0)=u0, u_t(x,0)=v0
def compute_pade_time2_22(u0_expr: sp.Expr, v0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], damping_op: Callable[[sp.Expr], sp.Expr] = None) -> tuple[sp.Expr, list[sp.Expr]]:
    damping_op = damping_op or (lambda f: sp.Integer(0))
    f0, f1 = u0_expr, v0_expr
    print(f" f0 = {f0}")
    print(f" f1 = {f1}")
    f2 = sp.simplify(pde_op(f0) - damping_op(f1))
    print(f" f2 = {f2}")
    f3 = sp.simplify(pde_op(f1) - damping_op(f2))
    print(f" f3 = {f3}")
    f4 = sp.simplify(pde_op(f2) - damping_op(f3))
    print(f" f4 = {f4}")

    denom_det = 2 * f1 * f3 - 3 * f2 ** 2
    try:
        a0 = sp.simplify(f0)
        a1 = sp.simplify(-(f0 * f1 * f4 - 2 * f0 * f2 * f3 - 4 * f1 ** 2 * f3 + 6 * f1 * f2 ** 2) / (2 * denom_det))
        a2 = sp.simplify((3 * f0 * f2 * f4 - 4 * f0 * f3 ** 2 - 6 * f1 ** 2 * f4 + 24 * f1 * f2 * f3 - 18 * f2 ** 3) / (12 * denom_det))
        b1 = sp.simplify(-(f1 * f4 - 2 * f2 * f3) / (2 * denom_det))
        b2 = sp.simplify((3 * f2 * f4 - 4 * f3 ** 2) / (12 * denom_det))
    except Exception:
        a0, a1, a2, b1, b2 = f0, sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0)
    print(f"  a0 = {a0}")
    print(f"  a1 = {a1}")
    print(f"  a2 = {a2}")
    print(f"  b1 = {b1}")
    print(f"  b2 = {b2}")

    numer = a0 + a1 * t_sym + a2 * t_sym ** 2
    denom = 1 + b1 * t_sym + b2 * t_sym ** 2
    P = numer / denom
    print(f"  P[2/2](x,t) = {P}")

    return P, [f0, f1, f2, f3, f4]

# Taylor-in-t recursion for u_tt = pde_op(u) - damping_op(u_t): f_k = d^k u/dt^k at t=0.
# f0, f1 are the ICs; differentiating the PDE k-2 times in t gives f_k = L(f_{k-2}) - D(f_{k-1}).
def _taylor_time2_coeffs(u0_expr: sp.Expr, v0_expr: sp.Expr, pde_op: Callable[[sp.Expr], sp.Expr], damping_op: Callable[[sp.Expr], sp.Expr], order: int = 6) -> list[sp.Expr]:
    f = [u0_expr, v0_expr]
    for k in range(2, order + 1):
        f.append(sp.simplify(pde_op(f[k - 2]) - damping_op(f[k - 1])))
    return f

# [3/3] Padé for 2nd-order-in-time PDEs: u_tt = pde_op(u) - damping_op(u_t), IC u(x,0)=u0, u_t(x,0)=v0
def compute_pade_time2_33(u0_expr: sp.Expr, v0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], damping_op: Callable[[sp.Expr], sp.Expr] = None) -> tuple[sp.Expr, list[sp.Expr]]:
    damping_op = damping_op or (lambda f: sp.Integer(0))
    f0, f1, f2, f3, f4, f5, f6 = _taylor_time2_coeffs(u0_expr, v0_expr, pde_op, damping_op, order=6)
    print(f" f0 = {f0}\n f1 = {f1}\n f2 = {f2}\n f3 = {f3}\n f4 = {f4}\n f5 = {f5}\n f6 = {f6}")

    c0, c1, c2, c3, c4, c5, c6 = f0, f1, f2/2, f3/6, f4/24, f5/120, f6/720

    D  = c3**3 - 2*c2*c3*c4 + c2**2*c5 + c1*c4**2 - c1*c3*c5
    D1 = -c3**2*c4 + c2*c4**2 + c2*c3*c5 - c2**2*c6 - c1*c4*c5 + c1*c3*c6
    D2 = -c3**2*c5 + c2*c3*c6 + c3*c4**2 - c2*c4*c5 - c1*c4*c6 + c1*c5**2
    D3 = -c3**2*c6 + 2*c3*c4*c5 + c2*c4*c6 - c2*c5**2 - c4**3

    if sp.simplify(D) == 0:
        raise ValueError("divide by zero error")

    b1, b2, b3 = sp.simplify(D1/D), sp.simplify(D2/D), sp.simplify(D3/D)
    print(f" b0=1 (normalised)\n b1={b1}\n b2={b2}\n b3={b3}")

    a0 = sp.simplify(c0)
    a1 = sp.simplify(c1 + c0*b1)
    a2 = sp.simplify(c2 + c1*b1 + c0*b2)
    a3 = sp.simplify(c3 + c2*b1 + c1*b2 + c0*b3)
    print(f" a0={a0}\n a1={a1}\n a2={a2}\n a3={a3}")

    numer = a0 + a1*t_sym + a2*t_sym**2 + a3*t_sym**3
    denom = 1 + b1*t_sym + b2*t_sym**2 + b3*t_sym**3
    P = numer / denom
    print(f" P[3/3](x,t) = {P}")
    return P, [f0, f1, f2, f3, f4, f5, f6]

# [4/2] Padé for 2nd-order-in-time PDEs: u_tt = pde_op(u) - damping_op(u_t), IC u(x,0)=u0, u_t(x,0)=v0
def compute_pade_time2_42(u0_expr: sp.Expr, v0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], damping_op: Callable[[sp.Expr], sp.Expr] = None) -> tuple[sp.Expr, list[sp.Expr]]:
    damping_op = damping_op or (lambda f: sp.Integer(0))
    f0, f1, f2, f3, f4, f5, f6 = _taylor_time2_coeffs(u0_expr, v0_expr, pde_op, damping_op, order=6)
    print(f" f0 = {f0}\n f1 = {f1}\n f2 = {f2}\n f3 = {f3}\n f4 = {f4}\n f5 = {f5}\n f6 = {f6}")

    Delta = sp.simplify(5*f4**2 - 4*f3*f5)
    if Delta == 0:
        raise ValueError("compute_pade_time2_42: Delta = 5 f4^2 - 4 f3 f5 vanishes identically, b1/b2 are undetermined")
    b1 = sp.simplify((2*f3*f6 - 3*f4*f5) / (3*Delta))
    b2 = sp.simplify((6*f5**2 - 5*f4*f6) / (30*Delta))
    print(f" b0 = 1  (normalised)\n b1 = {b1}\n b2 = {b2}")

    a0 = sp.simplify(f0)
    a1 = sp.simplify(f1 + b1*f0)
    a2 = sp.simplify(f2/2 + b1*f1 + b2*f0)
    a3 = sp.simplify(f3/6 + b1*(f2/2) + b2*f1)
    a4 = sp.simplify(f4/24 + b1*(f3/6) + b2*(f2/2))
    print(f" a0 = {a0}\n a1 = {a1}\n a2 = {a2}\n a3 = {a3}\n a4 = {a4}")

    numer = a0 + a1*t_sym + a2*t_sym**2 + a3*t_sym**3 + a4*t_sym**4
    denom = 1 + b1*t_sym + b2*t_sym**2
    P = numer / denom
    print(f" P[4/2](x,t) = {P}")
    return P, [f0, f1, f2, f3, f4, f5, f6]

# [2/4] Padé for 2nd-order-in-time PDEs: u_tt = pde_op(u) - damping_op(u_t), IC u(x,0)=u0, u_t(x,0)=v0
def compute_pade_time2_24(u0_expr: sp.Expr, v0_expr: sp.Expr, x_sym: sp.Symbol, t_sym: sp.Symbol, pde_op: Callable[[sp.Expr], sp.Expr], damping_op: Callable[[sp.Expr], sp.Expr] = None) -> tuple[sp.Expr, list[sp.Expr]]:
    damping_op = damping_op or (lambda f: sp.Integer(0))
    f0, f1, f2, f3, f4, f5, f6 = _taylor_time2_coeffs(u0_expr, v0_expr, pde_op, damping_op, order=6)
    print(f" f0 = {f0}\n f1 = {f1}\n f2 = {f2}\n f3 = {f3}\n f4 = {f4}\n f5 = {f5}\n f6 = {f6}")

    M = sp.Matrix([[3*f2,  6*f1,   6*f0,  0],
                   [4*f3, 12*f2,  24*f1, 24*f0],
                   [5*f4, 20*f3,  60*f2, 120*f1],
                   [6*f5, 30*f4, 120*f3, 360*f2]])
    if sp.simplify(M.det()) == 0:
        raise ValueError("compute_pade_time2_24: Toeplitz system is singular, b1..b4 are undetermined")
    b1, b2, b3, b4 = [sp.simplify(v) for v in M.LUsolve(sp.Matrix([-f3, -f4, -f5, -f6]))]
    print(f" b0 = 1  (normalised)\n b1 = {b1}\n b2 = {b2}\n b3 = {b3}\n b4 = {b4}")

    a0 = sp.simplify(f0)
    a1 = sp.simplify(f1 + b1*f0)
    a2 = sp.simplify(f2/2 + b1*f1 + b2*f0)
    print(f" a0 = {a0}\n a1 = {a1}\n a2 = {a2}")

    numer = a0 + a1*t_sym + a2*t_sym**2
    denom = 1 + b1*t_sym + b2*t_sym**2 + b3*t_sym**3 + b4*t_sym**4
    P = numer / denom
    print(f" P[2/4](x,t) = {P}")
    return P, [f0, f1, f2, f3, f4, f5, f6]

    return P_u, P_v, taylor_u, taylor_v