using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions

include("./hagerzhang.jl")
"""
# Solving `TrustRegionSubproblem`` via a given regularizer or a radius or radius-free 

## the strict regularization mode
strictly solve a quadratic regularization problems given a regularizer :λ 
## the radius free mode
compute the Lagrangian multiplier according to the adaptive :γ instead. see the paper for details.
## the strict radius mode
if you choose a radius, this function is a bisection procedure 
    following the complexity of O(loglog(1/ε)) by Ye
you may find other popular methods like 
  Steihaug-CG, Steihaug-Toint Lanczos method, et cetera,
  which are the most practical TRS solvers, as far as I know,
  you can find their implementations in GLTR and GLRT in GALAHAD library.  
"""
function TrustRegionSubproblem(
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


    if mode == :reg
        ######################################
        # the strict regularization mode
        ######################################
        # strictly solve a quadratic regularization problems
        #   given a regularizer :λ 
        ######################################
        alpha = -(_Q + λ .* _G) \ _c
        return alpha
    else
        eigvalues = eigvals(_Q, _G)
        sort!(eigvalues)
        lmin, lmax = eigvalues
        lb = max(1e-8, -lmin)
        ub = max(lb, lmax) + Δl
        if mode == :tr
            ######################################
            # the strict radius mode
            ######################################
            # strictly solve TR given a radius :Δ
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
        elseif mode == :free
            ######################################
            # the radius free mode
            ######################################
            # compute the Lagrangian multiplier 
            # according to the adaptive :γ instead.
            # see the paper for details.
            ######################################
            lb = max(0, -lmin)
            lmax = max(lb, lmax) + 1e3
            state.λ = state.γ * lmax + max(1 - state.γ, 0) * lb
            try
                alpha = -(_Q + state.λ .* _G) \ _c
                return alpha
            catch
                try
                    # this means it is singular
                    # for example, some direction 
                    # a wild solution
                    alpha = -(Diagonal(_Q) + state.λ .* Diagonal(_G)) \ _c
                    return alpha
                catch
                    println(_Q)
                    println(_G)
                    println(_c)
                    println(state.λ)
                    error("failed at evaluate Q")
                end
            end
            return alpha

        else
            ex = ErrorException("Only support :tr, :reg, and :free")
            throw(ex)
        end
    end
end

function TRStyleLineSearch(
    iter::IterationType,
    z::Tx,
    s::Tx,
    vHv::Real,
    vg::Real,
    f₀::Real,
    γ₀::Real=1.0;
    ρ::Real=0.7,
    ψ::Real=0.8
) where {IterationType,Tx}
    it = 1
    γ = γ₀
    fx = f₀
    while true
        # summarize
        x = z + s * γ
        fx = iter.f(x)
        dq = -γ^2 * vHv / 2 - γ * vg
        df = f₀ - fx
        ro = df / dq
        acc = (ro > ρ && df > 0) || it <= iter.itermax
        if !acc
            γ *= ψ
        else
            break
        end
        it += 1
    end
    return γ, fx, it
end

function HagerZhangLineSearch(
    iter::IterationType,
    gx, fx,
    x::Tx,
    s::Tx,
) where {IterationType,Tx}

    ϕ(α) = iter.f(x .+ α .* s)
    function dϕ(α)
        if iter.g !== nothing
            gv = iter.g(x + α .* s)
        else
            gv = similar(s)
            iter.ga(gv, x + α .* s)
        end
        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = iter.f(x .+ α .* s)
        if iter.g !== nothing
            gv = iter.g(x + α .* s)
        else
            gv = similar(s)
            iter.ga(gv, x + α .* s)
        end
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)
    lsa = HagerZhangEx()
    α, fx, it = lsa(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
    return α, fx, it
end

