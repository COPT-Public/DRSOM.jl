
using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions

"""
    DRSOMPlusIteration(; <keyword-arguments>)
"""
Base.@kwdef mutable struct DRSOMPlusIteration{Tx,Tf,Tr,Tg,TH,Tψ,Tc,Tt}
    f::Tf             # f: smooth f
    ψ::Tψ = nothing   # ψ: nonsmooth part (not implemented yet)
    rh::Tr = nothing  # hessian-vector product function to produce [g, Hg, Hd]
    g::Tg = nothing   # gradient function
    H::TH = nothing   # hessian function
    x0::Tx            # initial point
    cfg::Tc = nothing # gradient config
    tp::Tt = nothing  # gradient tape
    t::Dates.DateTime = Dates.now()
    itermax::Int64 = 20
    mode = :forward
    direction = :krylov
    direction_num::Int64 = 1
    ϵk::Float64 = 1e6 # when to use Krylov
end


Base.IteratorSize(::Type{<:DRSOMPlusIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct DRSOMPlusState{R,Tx,Tq,Tc}
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # gradient buffer (for temporary use)
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    α::Tx             # stepsizes for directions...
    Q::Tq             # Q for trs
    c::Tc             # c for trs
    Δ::R              # trs radius
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ϵ::R              # eps 2: residual for gradient 
    γ::R = 1e-16      # scaling parameter γ for λ
    λ::R = 1e-16      # dual λ
    it::Int = 1       # inner iteration #. for trs adjustment
    t::R = 0.0        # running time
    λ₁::Float64 = 0.0   # smallest curvature if available
end

function TrustRegionSubproblem(Q, c, state::DRSOMPlusState; G=diagmQ(ones(2)))
    try
        # for d it is too small, reduce to a Cauchy point ?
        eigvalues = eigvals(Q)
        sort!(eigvalues)
        lmin, lmax = eigvalues
        lb = max(0, -lmin)
        lmax = max(lb, lmax) + 1e4
        state.λ = state.γ * lmax + max(1 - state.γ, 0) * lb
        alpha = -(Q + state.λ .* G) \ c
        return alpha
    catch
        print(Q)
        print(state.λ)
    end
end


function Base.iterate(iter::DRSOMPlusIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    grad_f_x = similar(z)
    grad_f_b = similar(z)
    if iter.mode ∈ (:forward, false)
        Hg, _ = iter.rh(iter.f, grad_f_x, z - z, z; cfg=iter.cfg)
    elseif iter.mode ∈ (:backward, true)
        Hg, _ = iter.rh(grad_f_x, grad_f_b, z - z, z; tp=iter.tp)
    else
        grad_f_x = iter.g(z)
        H = iter.H(z)
        Hg = H * grad_f_x
    end
    Q11 = grad_f_x' * Hg
    ######################################
    c1 = -grad_f_x' * grad_f_x
    Q = [Q11 0; 0 0]
    c = [c1; 0]
    # now use a TRS to solve (gamma, alpha)
    ls = 1.0
    it = 1
    while true
        if Q11 > 0
            a1 = -c1 / Q11 * ls
            dq = -a1 * Q11 * a1 / 2 - a1 * c1
        else
            a1 = 0.1 * ls
            dq = -a1 * c1
        end
        y = z - a1 .* grad_f_x
        # ######################################
        # todo, eval proximal here?
        #   - we should design special alg. for proximal case.
        # evaluate TRS
        fx = iter.f(y)
        df = fz - fx
        ro = df / dq
        if ro > 0.1 || it == iter.itermax
            t = (Dates.now() - iter.t).value / 1e3
            d = y - z
            state = DRSOMPlusState(
                x=y,
                y=y,
                z=z,
                fx=fx,
                fz=fz,
                Q=Q,
                c=c,
                ∇f=grad_f_x,
                ∇fz=z,
                ∇fb=grad_f_b,
                α=[a1, 0],
                d=d,
                Δ=norm(d, 2),
                dq=dq,
                df=df,
                ρ=ro,
                ϵ=norm(grad_f_x, 2),
                γ=1e-6,
                λ=1e-6,
                it=it,
                t=t,
            )
            return state, state
        else
            ls *= 0.5
            it += 1
        end
    end
end

function generate_direction()
end

"""
Solve an iteration using TRS to produce stepsizes,
alpha: extrapolation
gamma: gradient step
"""
function Base.iterate(iter::DRSOMPlusIteration, state::DRSOMPlusState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    # construct trs
    # compute Hg, Hd first
    if iter.mode ∈ (:forward, false)
        Hg, Hd = iter.rh(iter.f, state; cfg=iter.cfg)
    elseif iter.mode ∈ (:backward, true)
        # todo, not ready yet.
        # compute gradient first
        Hg, Hd = iter.rh(state; tp=iter.tp)
    else
        state.∇f = iter.g(state.x)
        H = iter.H(state.x)
        Hg = H * state.∇f
        Hd = H * state.d
    end
    gnorm = norm(state.∇f)
    dnorm = norm(state.d)
    # add new directions v
    #   and again, we compute Hv
    if gnorm < iter.ϵk
        if iter.direction == :krylov
            # - krylov:
            vals, vecs, _ = KrylovKit.eigsolve(H, n, 1, :LR, Float64)
            v = reshape(vecs[1], n, 1)
            HD = [-Hg / gnorm Hd / dnorm H * v]
            D = [-state.∇f / gnorm state.d / dnorm v]
        elseif iter.direction == :homokrylov
            # - krylov:
            B = [H state.∇f; state.∇f' 0]
            vals, vecs, _ = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64)
            v = reshape(vecs[1][1:end-1], n, 1)
            v = v / norm(v)
            HD = [-Hg / gnorm H * v][:, 2]
            D = [-state.∇f / gnorm v][:, 2]

            state.λ₁ = vals[1]
        elseif iter.direction == :gaussian
            # - gaussian
            Σ = Diagonal(ones(n) .+ 0.01) - state.∇f * state.∇f' / gnorm^2 #- state.d * state.d' / dnorm^2
            D = MvNormal(zeros(n), Σ ./ iter.direction_num)
            # sanity check:

            V = rand(D, iter.direction_num)
            # normalization
            V = V * (1 ./ norm.(eachcol(V)) |> Diagonal)

            if iter.mode ∈ (:forward, false)
                throw(ErrorException("forward mode not supported yet"))
            elseif iter.mode ∈ (:backward, true)
                Hv = reduce(hcat, map(v -> iter.rh(state, v; tp=iter.tp), eachcol(V)))
            else
                Hv = H * V
            end

            HD = [-Hg / gnorm Hd / dnorm Hv]
            D = [-state.∇f / gnorm state.d / dnorm V]
        else
            HD = [-Hg / gnorm Hd / dnorm]
            D = [-state.∇f / gnorm state.d / dnorm]
        end
    else
        HD = [-Hg / gnorm Hd / dnorm]
        D = [-state.∇f / gnorm state.d / dnorm]
    end
    state.Q = Q = D' * HD
    state.c = c = D' * state.∇f
    G = D' * D
    it = 1
    while true
        alp = TrustRegionSubproblem(Q, c, state; G=G)
        x = y = state.z + D * alp
        fx = iter.f(x)
        dq = -alp' * Q * alp / 2 - alp' * c
        df = fz - fx
        ro = df / dq
        if (df < 0) || (ro <= 0.1)
            state.γ *= 5
        elseif ro >= 0.5
            state.γ = max(min(sqrt(state.γ), log(10, state.γ + 1)), 1e-16)
        end
        if (ro > 0.05 && df > 0) || it == iter.itermax
            state.α = alp
            state.x = x
            state.y = y
            state.fx = fx
            state.ρ = ro
            state.dq = dq
            state.df = df
            state.d = x - z
            state.Δ = sqrt(alp' * G * alp)
            state.it = it
            state.ϵ = norm(state.∇f)
            state.it = it
            state.t = (Dates.now() - iter.t).value / 1e3
            return state, state
        end
        it += 1
    end
end

drsom_stopping_criterion(tol, state::DRSOMPlusState) =
    (state.Δ <= tol / 1e2) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol

sprintarray(arr) = join(map(x -> @sprintf("%+.0e", x), arr), ",")
function drsom_display(it, state::DRSOMPlusState)
    if it == 1
        log = @sprintf("%5s | %10s | %13s | %7s | %7s | %5s | %5s | %6s | %2s | %6s |\n",
            "k", "f", "α ($(state.α |> length))", "Δ", "|∇f|", "γ", "λ", "ρ", "kₜ", "t",
        )
        format_header(log)
        @printf("%s", log)
    end
    if mod(it, 30) == 0
        @printf("%5s | %10s | %13s | %7s | %7s | %5s | %5s | %6s | %2s | %6s |\n",
            "k", "f", "α ($(state.α |> length))", "Δ", "|∇f|", "γ", "λ", "ρ", "kₜ", "t",
        )

    end
    @printf("%5d | %+.3e | %13s | %.1e | %.1e | %.0e | %.0e | %+.0e | %2d | %6.1f |\n",
        it, state.fx, sprintarray(state.α[1:min(2, end)]), state.Δ, state.ϵ, state.γ, state.λ, state.ρ, state.it, state.t
    )
end

default_solution(::DRSOMPlusIteration, state::DRSOMPlusState) = state.x
