using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions


function TrustRegionSubproblem(
    Q, c,
    state::StateType;
    G=diagm(ones(2)),
    mode=:free,
    Δ::Float64=0.0,
    Δϵ::Float64=1e-4,
    Δl::Float64=1e2,
    λ::Float64=0.0
) where {StateType}

    eigvalues = eigvals(Symmetric(Q))

    sort!(eigvalues)
    lmin, lmax = eigvalues
    lb = max(1e-8, -lmin)
    ub = max(lb, lmax) + Δl
    if mode == :reg
        # strictly solve a quadratic regularization problems
        #   given a regularizer :λ 
        alpha = -(Q + λ .* G) \ c
        return alpha

    elseif mode == :tr
        # strictly solve TR via
        #   given a radius :Δ
        # this is a bisection procedure
        λ = lb
        try
            alpha = -(Q + λ .* G) \ c
        catch
            println(lb)
            printstyled(Q)
            println(Q + λ .* G)
        end

        s = sqrt(alpha' * G * alpha) # size

        if s <= Δ
            # damped Newton step is OK
            state.λ = λ
            return alpha
        end
        # else we must hit the boundary
        while (ub - lb) > Δϵ
            λ = (lb + ub) / 2
            alpha = -(Q + λ .* G) \ c
            s = sqrt(alpha' * G * alpha) # size
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

    else
        ex = ErrorException("Only support :tr and :reg")
        throw(ex)
    end
end