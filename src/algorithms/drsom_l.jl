# DRSOM-L:
# @author: Chuwen Z.
#     a radius free, quasi-Newton DRSOM implementation 
#     referred to as the DRSOM with Hessian-Learnings (DRSOM-L).
# @note:
#     - in this scheme, we find an approximation of H
#         using SR1 and DFP like low rank updates
#         while preserving the negative curvature, if possible.
#     - in this view, the Hessian-vector products are no longer needed.

using Base.Iterators
using LinearAlgebra
using Printf
using Dates
using KrylovKit
using Distributions

"""
    DRSOMLIteration(; <keyword-arguments>)
"""
Base.@kwdef mutable struct DRSOMLIteration{Tx,Tf,Tg,Tψ,Tc,Tt,Tr}
    f::Tf             # f: smooth f
    ψ::Tψ = nothing   # ψ: nonsmooth part (not implemented yet)
    g::Tg = nothing   # gradient function
    x0::Tx            # initial point
    cfg::Tc = nothing # gradient config
    tp::Tt = nothing  # gradient tape
    t::Dates.DateTime = Dates.now()
    itermax::Int64 = 20
    mode = :forward
    direction = :krylov
    direction_num::Int64 = 1
    hessian = :sr1
    hessian_rank::Tr = :∞
end


Base.IteratorSize(::Type{<:DRSOMLIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct DRSOMLState{R,Tx,Tq,Tc,Tu,Tw}
    k::Int            # iterate #
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    U::Tu             # vectors of Hessian such that H = U⋅W⋅U' = ∑ w⋅u⋅u'
    W::Tw             # diagonals of Hessian 
    H::Tq             # complete form of Hessian
    up::Tx            #
    un::Tx            #
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
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
end

function TrustRegionSubproblem(Q, c, state::DRSOMLState; G=diagmQ(ones(2)))
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


function Base.iterate(iter::DRSOMLIteration)
    iter.t = Dates.now()
    z = copy(iter.x0)
    fz = iter.f(z)
    n = z |> length
    grad_f_x = similar(z)
    if iter.mode ∈ (:backward, true)
        ReverseDiff.gradient!(grad_f_x, iter.tp, z)
    else
        throw(ErrorException("not implemented"))
    end

    ######################################
    Q11 = 1.0
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
            state = DRSOMLState(
                x=y,
                y=y,
                z=z,
                fx=fx,
                fz=fz,
                Q=Q,
                c=c,
                U=zeros(n, 0),
                up=zeros(n),
                un=zeros(n),
                W=zeros(0),
                H=zeros(n, n), # Array{Float64}(undef, n, n)
                ∇f=grad_f_x,
                ∇fz=similar(grad_f_x),
                α=[a1],
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
                k=1,
            )
            return state, state
        else
            ls *= 0.05
            it += 1
        end
    end
end

"""
Solve an iteration using TRS to produce stepsizes,
alpha: extrapolation
gamma: gradient step
"""
function Base.iterate(iter::DRSOMLIteration, state::DRSOMLState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    copy!(state.∇fz, state.∇f)
    # construct trs
    # compute Hg, Hd first
    if iter.mode ∈ (:backward, true)
        ReverseDiff.gradient!(state.∇f, iter.tp, state.x)
    else
        throw(ErrorException("not implemented"))
    end
    gnorm = norm(state.∇f)
    dnorm = norm(state.d)


    # update approx. Hessian
    if iter.hessian == :sr1
        Δg = state.∇f - state.∇fz
        Δy = state.H * state.d
        # new hessian update vector
        u = Δg - Δy
        w = (state.d' * (u))
        # not orthogonal
        if w != 0
            w = 1 / w
            if iter.hessian_rank ∉ (:∞, -1) && iter.hessian_rank > 0
                # preserving Hessian below some rank
                #   rank(H) <= iter.hessian_rank
                state.U = [state.U u][:, 1:min(end, iter.hessian_rank)]
                state.W = [state.W..., w][1:min(end, iter.hessian_rank)]
            else
                # else allow potentially full rank
                state.H += w * u * u'
            end
        end
    elseif iter.hessian == :bfgs
        if iter.hessian_rank ∈ (:∞, -1)
            Δg = state.∇f - state.∇fz
            # new hessian update vector
            Δy = state.H * state.d
            w = (state.d' * (Δg))
            (w != 0) && (state.H += Δg * Δg' / w)
            w = -(state.d' * Δy)
            (w != 0) && (state.H += Δy * Δy' / w)
        else
            throw(ErrorException("BFGS do not support limited rank H approx."))
        end
    else
        iter.hessian == :sr1p
        Δg = state.∇f - state.∇fz
        Δy = state.H * state.d
        # new hessian update vector
        u = Δg - Δy
        w = (state.d' * (u))

        # not orthogonal
        if w != 0
            w = 1 / w
            if iter.hessian_rank ∉ (:∞, -1) && iter.hessian_rank > 0
                # preserving Hessian below some rank
                #   rank(H) <= iter.hessian_rank
                state.U = [state.U u][:, 1:min(end, iter.hessian_rank)]
                state.W = [state.W..., w][1:min(end, iter.hessian_rank)]
                if w > 0
                    state.up = 1 / state.k * (state.up * (state.k - 1) + u * sqrt(w))
                else
                    state.un = 1 / state.k * (state.un * (state.k - 1) + u * sqrt(-w))
                end
            else
                # else allow potentially full rank
                state.H += w * u * u'
            end
        end
    end
    D = [-state.∇f / gnorm state.d / dnorm]
    # add new directions v
    #   and again, we compute Hv
    if iter.direction == :krylov
        #     # - krylov:
        #     vals, vecs, _ = KrylovKit.eigsolve(H, n, 1, :SR, Float64)
        #     v = reshape(vecs[1], n, 1)
        #     HD = [-Hg / gnorm Hd / dnorm H * v]
        #     D = [-state.∇f / gnorm state.d / dnorm v]

        # elseif iter.direction == :gaussian
        #     # - gaussian
        #     Σ = Diagonal(ones(n) .+ 0.01) - state.∇f * state.∇f' / gnorm^2 #- state.d * state.d' / dnorm^2
        #     D = MvNormal(zeros(n), Σ ./ iter.direction_num)
        #     # sanity check:

        #     V = rand(D, iter.direction_num)
        #     # normalization
        #     V = V * (1 ./ norm.(eachcol(V)) |> Diagonal)

        #     if iter.mode ∈ (:forward, false)
        #         throw(ErrorException("forward mode not supported yet"))
        #     elseif iter.mode ∈ (:backward, true)
        #         Hv = reduce(hcat, map(v -> iter.rh(state, v; tp=iter.tp), eachcol(V)))
        #     else
        #         Hv = H * V
        #     end

        #     HD = [-Hg / gnorm Hd / dnorm Hv]
        #     D = [-state.∇f / gnorm state.d / dnorm V]
        # else
        #     throw(ErrorException(@sprintf("unknown direction %s", iter.direction)))
    end
    # compute Q,c,G
    # use low-rank updates
    if iter.hessian_rank ∈ (:∞, -1)
        state.Q = Q = D' * state.H * D
    elseif iter.hessian_rank > 0
        UD = D' * state.U
        upD = D' * state.up
        unD = D' * state.un
        state.Q = Q = UD * Diagonal(state.W) * UD' + upD * upD' - unD * unD'
    else

    end
    state.c = c = D' * state.∇f
    G = D' * D
    # start inner iterations

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
            state.k += 1
            return state, state
        end
        it += 1
    end
end

drsom_stopping_criterion(tol, state::DRSOMLState) =
    (state.Δ <= tol / 1e2) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol

function drsom_display(it, state::DRSOMLState)
    if it == 1
        log = @sprintf("%5s | %10s | %13s | %7s | %7s | %5s | %5s | %6s | %2s | %6s \n",
            "k", "f", string(repeat(" ", 7 * (state.α |> length) - 2), "α"), "Δ", "|∇f|", "γ", "λ", "ρ", "it", "t",
        )
        format_header(log)
        @printf("%s", log)
    end
    if mod(it, 30) == 0
        @printf("%5s | %10s | %13s | %7s | %7s | %5s | %5s | %6s | %2s | %6s \n",
            "k", "f", string(repeat(" ", 7 * (state.α |> length) - 2), "α"), "Δ", "|∇f|", "γ", "λ", "ρ", "it", "t",
        )

    end
    @printf("%5d | %+.3e | %13s | %.1e | %.1e | %.0e | %.0e | %+.0e | %2d | %6.1f \n",
        it, state.fx, sprintarray(state.α), state.Δ, state.ϵ, state.γ, state.λ, state.ρ, state.it, state.t
    )
end

default_solution(::DRSOMLIteration, state::DRSOMLState) = state.x
