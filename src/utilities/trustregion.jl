using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit



@doc raw"""
A simple procedure to solve TRS via a given regularizer 
This is only used in DRSOM

- the radius free mode: compute the Lagrangian multiplier according to the adaptive ``\gamma`` instead. 
    See the paper for details.
- the strict regularization mode: strictly solve a quadratic regularization problems given a regularizer ``\lambda`` 

- the strict radius mode: if you choose a radius, this function is a bisection procedure 
    following the complexity of ``O(\log\log(1/ε))`` by Ye

We use this function (radius-free mode) `only` in DRSOM, since the eigenvalues are easy to solve. 
"""
function SimpleTrustRegionSubproblem(
    Q,
    c::Vector{Float64},
    state::StateType;
    G=diagm(ones(2)),
    mode=:free,
    Δ::Float64=0.0,
    Δϵ::Float64=1e-4,
    Δl::Float64=1e2,
    λ::Float64=0.0
) where {StateType}

    # the dimension of Q and G should not be two big.
    _Q, _G = Symmetric(Q), Symmetric(G)
    _c = c

    if (mode == :free) && (length(_c) == 2)
        ######################################
        # the radius free mode
        ######################################
        # @special treatment for d = 2 (DRSOM 2D)
        # when it is small, 
        #   use more simple ways to calculate eigvalues.
        a = _Q[1, 1]
        b = _Q[1, 2]
        d = _Q[2, 2]
        t = a + d
        s = a * d - b^2
        lmin = t / 2 - (t^2 / 4 - s)^0.5
        lmax = t / 2 + (t^2 / 4 - s)^0.5
        # eigvalues = eigvals(_Q, _G)
        # sort!(eigvalues)
        # lmin, lmax = eigvalues
        lb = max(1e-8, -lmin)
        ub = max(lb, lmax) + Δl
        state.λ = state.γ * lmax + max(1 - state.γ, 0) * lb
        _QG = _Q + state.λ .* _G
        a = _QG[1, 1]
        b = _QG[1, 2]
        d = _QG[2, 2]
        s = a * d - b^2
        K = [d/s -b/s; -b/s a/s]
        alpha = -K * _c
        return alpha
    end

    ######################################
    # compute eigenvalues.
    ######################################
    eigvalues = begin
        try
            eigvals(_Q + 1e-10I, _G)
        catch e
            eigvals(_Q + 1e1I, _G)
        end
    end
    sort!(eigvalues)
    lmin, lmax = eigvalues[1], eigvalues[end]
    lb = max(1e-15, -lmin)
    ub = max(lb, lmax) + Δl
    if mode == :reg
        ######################################
        # the strict regularization mode
        ######################################
        # strictly solve a quadratic regularization problems
        #   given a regularizer :λ 
        ######################################
        state.λ = λ = state.γ * lmax + max(1 - state.γ, 0) * lb
        alpha = -(_Q + λ .* _G) \ _c
        return alpha
    end
    if mode == :trbisect
        ######################################
        # the strict radius mode
        ######################################
        # strictly solve TR given a radius :Δ 
        #   via bisection
        ######################################

        λ = lb
        try
            alpha = -(_Q + λ .* _G) \ _c
        catch e
            println(lb)
            printstyled(_Q)
            println(_Q + λ .* _G)
            throw(e)
        end

        s = sqrt(alpha' * _G * alpha) # size

        if s <= Δ
            # damped Newton step is OK
            state.λ = λ
            return alpha
        end
        # else we must hit the boundary
        while (ub - lb) > Δϵ
            λ = (lb + ub) / 2
            alpha = -(_Q + λ .* _G) \ _c
            s = sqrt(alpha' * _G * alpha) # size
            if s > Δ + Δϵ
                lb = λ
            elseif s < Δ - Δϵ
                ub = λ
            else
                # good enough
                break
            end
        end
        state.λ = λ
        return alpha
    end
    ex = ErrorException("Do not support the options $mode")
    throw(ex)
end





@doc raw"""
A direct procedure to solve `TRS` utilizing Cholesky factorization

This is a slight revised version of standard Newton's method
using a hybrid bisection + regularized Newton's approach

```math
\begin{aligned}
\min ~& 1/2 x^TQx + c^Tx \\
\text{s.t.} ~& \|x\| \le \Delta
\end{aligned}
```
"""
function TrustRegionCholesky(
    Q,
    c::Vector{Float64},
    Δ::Float64=0.0;
    Δϵ::Float64=1e-2,
    λₗ=0.0,
    λᵤ=Inf
)
    F = cholesky(Q + λₗ * I, check=false)
    if issuccess(F) # meaning psd
        x = F \ (-c)
        if norm(x) < Δϵ + Δ
            return x, 0, 0.0, 0
        end
    end
    # else it is indefinite.
    # mild estimate to ensure p.s.d
    if λᵤ < Inf
        λₖ = (λₗ + λᵤ) / 2
    else
        λₖ = ((sum(abs, Q - Diagonal(Q), dims=1)[:] + abs.(diag(Q))) |> maximum) / 2
    end
    k = 1
    bool_indef = false
    p = nothing
    while true
        F = cholesky(Q + λₖ * I, check=false, perm=p)
        if !issuccess(F)
            bool_indef = true
            λₖ *= 1.05
            continue
        end
        p === nothing ? F.p : p
        if bool_indef
            λₗ = λₖ / 1.02
        end
        bool_indef = false

        Rt = F.L
        x = F \ (-c)
        q_l = Rt \ x
        norm2_s = dot(x, x)

        l2 = dot(q_l, q_l) / sqrt(norm2_s)^3
        l1 = -(sqrt(norm2_s) - Δ) / sqrt(norm2_s) / Δ

        # @info "this" (-l1 / l2) (norm2_s * (sqrt(norm2_s) - Δ) / (Δ * dot(q_l, q_l)))

        dℓ = -(l1 / (l2 + 1e-2))
        if dℓ < 0
            λᵤ = min(λᵤ, λₖ)
            α = min(0.9995 * ((λₖ - λₗ) / -dℓ), 1.0)
        else
            λₗ = max(λₗ, λₖ)
            α = min(0.9995 * ((λᵤ - λₖ) / dℓ), 1.0)
        end

        if (α <= 0.8) && (λᵤ < Inf)
            λₖ = (λᵤ + λₗ) / 2
        else
            λₖ += α * dℓ
        end
        @debug """iterate""" k, [λₗ, λᵤ], λₖ, α, dℓ, norm2_s, (Δ^2)


        k += 1
        if (abs(l1) < Δϵ) || (k > 20) || ((λᵤ - λₗ) < Δϵ)
            break
        end
    end
    return x, λₖ, -λₗ, k, p
end