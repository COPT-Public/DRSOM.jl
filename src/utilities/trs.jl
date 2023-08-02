using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions

include("./hagerzhang.jl")
include("./backtracking.jl")

const lsa::HagerZhangEx = HagerZhangEx(
# linesearchmax=10
)
const lsb::BackTrackingEx = BackTrackingEx(
    ρ_hi=0.8,
    ρ_lo=0.1,
    order=2
)

@doc raw"""
A simple procedure to solve TRS via a given regularizer || a radius 

- the radius free mode: compute the Lagrangian multiplier according to the adaptive ``\gamma`` instead. 
    See the paper for details.
- the strict regularization mode: strictly solve a quadratic regularization problems given a regularizer ``\lambda`` 

- the strict radius mode: if you choose a radius, this function is a bisection procedure 
    following the complexity of ``O(\log\log(1/ε))`` by Ye

We use this function (radius-free mode) `only` in DRSOM, since the eigenvalues are easy to solve. 
For full-dimensional problems, you should use `TrustRegionSubproblem` or other popular methods like 
  Steihaug-CG, Steihaug-Toint Lanczos method, et cetera,
  which are more practical, as far as I know,
  you can find their implementations in GLTR and GLRT in GALAHAD library.  
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
    if mode == :reg
        ######################################
        # the strict regularization mode
        ######################################
        # strictly solve a quadratic regularization problems
        #   given a regularizer :λ 
        ######################################
        alpha = -(_Q + λ .* _G) \ _c
        return alpha
    end
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
    eigvalues = eigvals(_Q, _G)
    sort!(eigvalues)
    lmin, lmax = eigvalues
    lb = max(1e-8, -lmin)
    ub = max(lb, lmax) + Δl

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

function TRStyleLineSearch(
    iter::IterationType,
    z::Tx,
    s::Tg,
    vHv::Real,
    vg::Real,
    f₀::Real,
    γ₀::Real=1.0;
    ρ::Real=0.7,
    ψ::Real=0.8
) where {IterationType,Tx,Tg}
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


function BacktrackLineSearch(
    iter::IterationType,
    gx, fx,
    x::Tx,
    s::Tg
) where {IterationType,Tx,Tg}

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

    try
        α, fx, it = lsb(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        return α, fx, it
    catch y
        isa(y, LineSearchException) # && println() # todo

        return 0.1, fx, 1
    end
end

function BacktrackLineSearch(
    f, g,
    gx, fx,
    x::Tx,
    s::Tg
) where {Tx,Tg}

    ϕ(α) = f(x .+ α .* s)
    function dϕ(α)

        gv = g(x + α .* s)

        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = f(x .+ α .* s)
        gv = g(x + α .* s)
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)
    try
        α, fx = lsb(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        return α, fx
    catch y
        # throw(y)
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end

end


function HagerZhangLineSearch(
    iter::IterationType,
    gx, fx,
    x::Tx,
    s::Tg;
    α₀::R=1.0
) where {IterationType,Tx,Tg,R<:Real}

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

    try
        α, fx, it = lsa(ϕ, dϕ, ϕdϕ, α₀, fx, dϕ_0)
        return α, fx, it
    catch y
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end
end

function HagerZhangLineSearch(
    f, g,
    gx, fx,
    x::Tx,
    s::Tg;
    α₀::R=1.0
) where {Tx,Tg,R<:Real}

    ϕ(α) = f(x .+ α .* s)
    function dϕ(α)

        gv = g(x + α .* s)

        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = f(x .+ α .* s)
        gv = g(x + α .* s)
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)
    try
        α, fx, it = lsa(ϕ, dϕ, ϕdϕ, α₀, fx, dϕ_0)
        return α, fx, it
    catch y
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end

end




@doc raw"""
A direct/indirect procedure to solve `TRS`.
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
    end
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
    eigvalues = eigvals(_Q, _G)
    sort!(eigvalues)
    lmin, lmax = eigvalues
    lb = max(1e-8, -lmin)
    ub = max(lb, lmax) + Δl

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