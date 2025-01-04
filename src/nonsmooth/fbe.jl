####################################################################################################
# basic tools for forward-backward envelope
####################################################################################################
# @reference: 
# --------------------------------------------------------------------------------------------------
# [1].  Stella, L., Themelis, A., Sopasakis, P., Patrinos, P.: 
#   A simple and efficient algorithm for nonlinear model predictive control. 
#   In: 2017 IEEE 56th Annual Conference on Decision and Control (CDC). pp. 1939–1944. IEEE (2017)
# [2]  Stella, L., Themelis, A., Patrinos, P.: 
#   Forward–backward quasi-Newton methods for nonsmooth optimization problems. 
#   Computational Optimization and Applications. 67, 443–487 (2017)
# [3].  Themelis, A., Stella, L., Patrinos, P.: 
#   Forward-Backward Envelope for the Sum of Two Nonconvex Functions: Further Properties and Nonmonotone Linesearch Algorithms. 
#   SIAM J. Optim. 28, 2274–2303 (2018). https://doi.org/10.1137/16M1080240
# --------------------------------------------------------------------------------------------------
# @note:
#   we follow the notations in [2] for the most part.
#   aliases:
#   [2] - MINFBE
#   [3] - ZeroFPR
####################################################################################################
using ProximalOperators: prox, prox!
# compute T and R, default to evaluate current x
function compute_T_R(iter, ℓ, x)
    # compute gradient (forward) point
    ∇ = iter.g(x)
    y = x - ℓ * ∇
    # compute proximal (backward) step to z
    T, h = prox(iter.h, y, ℓ)
    r = 1 / ℓ * (x - T)
    return ∇, y, T, r
end

@doc raw"""
```math 
$\varphi_\gamma(x)=f(x)+g\left(T_\gamma(x)\right)-\gamma\left\langle\nabla f(x), R_\gamma(x)\right\rangle+\frac{\gamma}{2}\left\|R_\gamma(x)\right\|^2$
```
"""
function ϕᵧ(iter, ℓ, x; β=0.0)
    # update the forward-backward envelope
    # at the current point x (not necessarily x)
    # the forward point             save as y
    # the gradient of trial x       save as ∇
    ∇, y, T, r = compute_T_R(iter, ℓ, x)
    # compute ϕ 
    ϕ = (
        iter.h(T) + iter.f(x) -
        ∇' * r * ℓ +
        0.5 * (1 - β) * ℓ * r' * r
    )
    return ϕ
end

function ϕᵧ∇φᵧ(iter, ℓ, x; β=0.0)
    # must call before take gradient 
    # so that T, r is updated
    ∇, y, T, r = compute_T_R(iter, ℓ, x)
    ∇ = r - ℓ * iter.hvp(x, r)
    # compute ϕ 
    ϕ = (
        iter.h(T) + iter.f(x) -
        ∇' * r * ℓ +
        0.5 * (1 - β) * ℓ * r' * r
    )
    return ∇, ϕ
end
CONSTANT_NON_DIAG_MAT = [
    [0.0; 1.0],
    [1.0; 0.0],
    [0.2; 0.7],
    [0.5; 1.0],
    [1.0; 0.9],
    [0.2; 0.85]
]
CONSTANT_DIAG_MAT = [
    [0.0; 0.1],
    [0.0; 0.15],
    [0.0; 0.2],
    [0.0; 0.5],
    [0.85; 0],
    [0.92; 0],
    [0.99; 0]
]
@doc raw"""
    interpolate for the FBE case.
    interpolate a smooth quadratic approximation of ϕ
"""
function directional_interpolation_fbe(iter, state; ϵₚ=1e-4, verbose=false, diag=false)
    dₙ = norm(state.d)

    # use fbe as the interpolation function instead
    ∇, y, T, r = compute_T_R(iter, state.ℓ, state.x)
    ∇ₓ, ϕx = ϕᵧ∇φᵧ(iter, state.ℓ, state.x; β=0.0)
    ϕ = (x) -> ϕᵧ(iter, state.ℓ, state.x; β=0.0)

    # do interpolation
    gₙ = norm(∇ₓ)
    V = [-∇ₓ ./ gₙ, state.d / dₙ]
    state.c = c = [-∇ₓ'∇ₓ / gₙ; ∇ₓ'state.d / dₙ]
    m = length(c)
    l = m * (m + 1) |> Int
    a = max(1e-4, ϵₚ * dₙ) * (diag ? CONSTANT_DIAG_MAT : CONSTANT_NON_DIAG_MAT)

    A = hcat([build_natural_basis(z) for z in a]...)'
    d = [c' * z for z in a]
    # trial points added to x
    x_n(_a) = (map((x, y) -> x * y, V, _a) |> sum) + state.x
    xs = [x_n(_a) for _a in a]
    b = [(x |> ϕ) - ϕx for x in xs] # rhs
    q = (A' * A + 1e-8I) \ (A' * (b - d))
    Q = Symmetric(
        [i <= j ? q[Int(j * (j - 1) / 2)+i] : 0
         for i = 1:m, j = 1:m], :U
    )
    state.Q = Q
    gg = 1.0
    gd = ∇ₓ' * state.d ./ gₙ ./ dₙ
    dd = 1.0
    state.G = Symmetric([gg -gd; -gd dd])

    # feed what is necessary
    state.ϕ = ϕx
    state.T = T
    state.r = r
    state.∇ϕ = ∇ₓ
    if !verbose
        return gₙ, dₙ, state.Q, state.c, state.G
    else
        return gₙ, dₙ, state.Q, state.c, state.G, A, a, b, d, q
    end
end