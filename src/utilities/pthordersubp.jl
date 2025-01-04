# this is a subproblem for the p-th order trust region subproblem
using Base.Iterators
using LinearAlgebra, SparseArrays
using Printf, Dates
using KrylovKit

@doc raw"""
A direct procedure to solve ``Cubic Regularized Subproblem``
```math
\begin{aligned}
\min ~& 1/2 x'Qx + c'x + M/6 ||x||^3
\end{aligned}
```
```math
equivalent to find the dual problem λ:
ψ(λ) = \min_x 1/2 c'x + 1/2 x'(Q+λI)x - 2/(3M^2)|λ|^3
```
!!!This is a preliminary version, no treatment for hard case.
"""
function CubicSubpCholesky(
    Q::Union{SparseMatrixCSC,Symmetric{R,SparseMatrixCSC{R,Int}}},
    c::Vector{R},
    M::R;
    Δϵ::R=1e-8, # tolerance for the step size
    λₗ=0.0,
    λᵤ=Inf
) where {R}
    @debug "c: $c"
    @debug "M: $M"
    @debug "initial λₗ: $λₗ λᵤ: $λᵤ"
    # else it is indefinite or too large step.
    # mild estimate to ensure p.s.d
    if λᵤ < Inf
        λₖ = (λₗ + λᵤ) / 2
    else
        # use diagonal dominance
        λₖ = ((sum(abs, Q - Diagonal(Q), dims=1)[:] + abs.(diag(Q))) |> maximum) / 2
        λᵤ = sqrt(norm(c) * M / 2) + λₖ
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

        x = F \ (-c)
        norm2_s = norm(x)

        dℓ = norm2_s - 2 * λₖ / M

        @debug """iterate""" k, [λₗ, λᵤ], (2 * λₖ / M), norm2_s, dℓ
        if dℓ < 0
            λᵤ = min(λᵤ, λₖ)
        else
            λₗ = max(λₗ, λₖ)
        end
        λₖ = (λᵤ + λₗ) / 2

        k += 1
        if (abs(dℓ) < Δϵ * norm2_s) || (k > 20) || ((λᵤ - λₗ) < Δϵ)
            return x, λₖ, -λₗ, k, p
        end
    end
end