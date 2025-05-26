module TRS
using Base.Iterators
using LinearAlgebra, SparseArrays
using Printf, Dates
using KrylovKit

include("./trustregionlr.jl")

@doc raw"""
A direct procedure to solve `TRS` utilizing Cholesky factorization

This is a slight revised version of standard Newton's method
using a hybrid bisection + regularized Newton's approach

```math
\begin{aligned}
\min ~& 1/2 x'Qx + c'x \\
\text{s.t.} ~& \|x\| \le \Delta
\end{aligned}
```
"""
function TrustRegionCholesky(
    Q::Union{SparseMatrixCSC,Symmetric{R,SparseMatrixCSC{R,Int}}},
    c::Vector{R},
    Δ::R=0.0;
    Δϵ::R=1e-2, # tolerance for the step size
    λₗ=0.0,
    λᵤ=Inf
) where {R}
    @debug "initial λₗ: $λₗ λᵤ: $λᵤ"
    F = cholesky(Q + λₗ * I, check=false)
    if issuccess(F) # meaning psd
        @debug """use provided λₗ cholesky successful
        """
        x = F \ (-c)
        if norm(x) < Δϵ + Δ
            @debug """use provided λₗ successful (|d| < Δ)
            """
            return x, 0, 0.0, 0
        end
    end
    @debug """use provided λₗ failed 
    """
    # else it is indefinite or too large step.
    # mild estimate to ensure p.s.d
    if λᵤ < Inf
        λₖ = (λₗ + λᵤ) / 2
    else
        # use diagonal dominance
        λₖ = ((sum(abs, Q - Diagonal(Q), dims=1)[:] + abs.(diag(Q))) |> maximum) / 2
    end
    k = 1
    bool_indef = false
    p = nothing
    @debug """start smart bisection
    initial λₖ: $λₖ
    """
    while true
        F = cholesky(Q + λₖ * I; check=false, perm=p)
        @debug "cholesky: $(issuccess(F))"
        if !issuccess(F)
            bool_indef = true
            λₖ = max(λₖ * 1.05, 1e-10)
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
        norm2_s = norm(x)

        l2 = dot(q_l, q_l) / norm2_s^3
        l1 = -(norm2_s - Δ) / norm2_s / Δ

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
        @debug """iterate""" k, [λₗ, λᵤ], λₖ, α, dℓ, norm2_s, Δ


        k += 1
        if (abs(l1) < Δϵ) || (k > 20) || ((λᵤ - λₗ) < Δϵ)
            break
        end
    end
    return x, λₖ, -λₗ, k, p
end


# ------------------------------------------------------------------------------
# Wrapper for Adachi's method; very slow, not recommended.
# ------------------------------------------------------------------------------
using ..ATRS

@doc raw"""
A direct procedure to solve `TRS` utilizing Adachi's method
"""
function AdachiTrustRegionSubproblem(
    Q,
    c::Vector{Float64},
    Δ::R=0.0;
    Δϵ::R=1e-2, # tolerance for the step size
) where {R}
    x, info = ATRS.trs(Q, c, Δ; tol=Δϵ)
    @debug """
    AdachiTrustRegionSubproblem
    |d|: $(x |> norm)
    λ: $(info.λ)
    """
    return x[:], info.λ[1], -0.0, 0.0, nothing
end


export SimpleTrustRegionSubproblem, TrustRegionCholesky, AdachiTrustRegionSubproblem
end # module TRS