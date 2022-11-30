
using Base.Iterators
using LinearAlgebra
using Printf
using Dates

"""
    DRSOMIteration(; <keyword-arguments>)
"""



Base.@kwdef mutable struct DRSOMIteration{Tx,Tf,Tr,Tg,TH,Tψ,Tc,Tt}
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
end


Base.IteratorSize(::Type{<:DRSOMIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct DRSOMState{R,Tx,Tq,Tc}
    x::Tx             # iterate
    fx::R             # new value f at x: x(k)
    fz::R             # old value f at z: x(k-1)
    ∇f::Tx            # gradient of f at x
    ∇fz::Tx           # gradient of f at z
    ∇fb::Tx           # gradient buffer (for temporary use)
    y::Tx             # forward point
    z::Tx             # previous point
    d::Tx             # momentum/fixed-point diff at iterate (= x - z)
    a1::R             # stepsize 1 parameter of gradient
    a2::R             # stepsize 2 parameter of momentum
    Q::Tq             # Q for trs
    c::Tc             # c for trs
    Δ::R              # trs radius
    dq::R             # decrease of estimated quadratic model
    df::R             # decrease of the real function value
    ρ::R              # trs descrease ratio: ρ = df/dq
    ϵ::R              # eps: residual for gradient 
    γ::R = 1e-16      # scaling parameter γ for λ
    ψ::R = 1.0        # 1-D linear search iteration #. if there is only one direction  
    λ::R = 1e-16      # dual λ
    it::Int = 1       # inner iteration #. for trs adjustment
    t::R = 0.0        # running time
end


function Base.iterate(iter::DRSOMIteration)
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
        H = LinearAlgebra.I # todo, change to interpolation
        Hg = H * grad_f_x
    end
    ######################################
    Q11 = grad_f_x' * Hg
    c1 = -grad_f_x' * grad_f_x
    # now use a LS to find the first iterate
    a2 = 0.0
    ls = 1.0
    it = 1
    while true
        a1 = ls # Q11 > 0 ? -c1 / Q11 * ls : ls
        dq = -a1 * Q11 * a1 / 2 - a1 * c1
        y = z - a1 .* grad_f_x
        # ######################################
        # todo, eval proximal here?
        #   - we should design special alg. for proximal case.
        # evaluate TRS
        fx = iter.f(y)
        df = fz - fx
        ro = df / dq
        if (df > 0) || it == iter.itermax
            t = (Dates.now() - iter.t).value / 1e3
            d = y - z
            state = DRSOMState(
                x=y,
                y=y,
                z=z,
                fx=fx,
                fz=fz,
                Q=Matrix{Float64}(undef, (1, 1)),
                c=[c1],
                ∇f=grad_f_x,
                ∇fz=z,
                ∇fb=grad_f_b,
                a1=a1,
                a2=a2,
                d=d,
                Δ=a1,
                dq=dq,
                df=df,
                ρ=ro,
                ϵ=norm(grad_f_x),
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


"""
Solve an iteration using TRS to produce stepsizes,
alpha: extrapolation
gamma: gradient step
"""
function Base.iterate(iter::DRSOMIteration, state::DRSOMState{R,Tx}) where {R,Tx}

    n = length(state.x)
    state.z = z = state.x
    state.fz = fz = state.fx
    state.∇fz = state.∇f
    # construct trs
    if iter.mode == :direct
        # use Hvp to calculate Q
        state.∇f = iter.g(state.x)
        gnorm = state.ϵ = norm(state.∇f)
        dnorm = norm(state.d)
        c1 = -state.∇f'state.∇f
        c2 = state.∇f'state.d
        state.c = c = [c1 / gnorm; c2 / dnorm]
        state.Q = Q = simple_itp(iter, state, [-state.∇f / gnorm, state.d / dnorm], c)
    else
        if iter.mode ∈ (:forward, false)
            Hg, Hd = iter.rh(iter.f, state; cfg=iter.cfg)
        elseif iter.mode ∈ (:backward, true)
            Hg, Hd = iter.rh(state; tp=iter.tp)
        elseif iter.mode ∈ (:directhess)
            H = iter.H(state.x)
            Hg = H * state.∇f
            Hd = H * state.d
        end

        gnorm = state.ϵ = norm(state.∇f)
        dnorm = norm(state.d)

        c1 = -state.∇f'state.∇f / gnorm
        c2 = state.∇f'state.d / dnorm
        state.c = c = [c1; c2]

        Q11 = state.∇f' * Hg / gnorm^2
        Q12 = -state.∇f' * Hd / gnorm / dnorm
        Q22 = state.d' * Hd / dnorm^2
        state.Q = Q = [Q11 Q12; Q12 Q22]
    end


    if gnorm <= 1e-10
        return state, state
    end

    gg = state.∇f' * state.∇f / gnorm^2
    gd = state.∇f' * state.d / gnorm / dnorm
    dd = state.d' * state.d / dnorm^2
    G = [gg -gd; -gd dd]

    # if the directions are too close (degenerate)
    #   you can do QR to find a proper subset of directions
    #   (but too expensive>)
    # we simply use the gradient
    bool_tr = abs.(tril(G, -1)) |> maximum < 0.97
    # start inner iterations
    it = 1
    if bool_tr
        while true

            a1, a2 = TrustRegionSubproblem(Q, c, state; G=G, mode=:free)

            x = y = state.z - a1 .* state.∇f / gnorm + a2 .* state.d / dnorm
            fx = iter.f(x)
            alp = [a1; a2]
            dq = -alp' * Q * alp / 2 - alp' * c
            df = fz - fx
            ro = df / dq
            if (df < 0) || (ro <= 0.1)
                state.γ *= 5
            elseif ro >= 0.5
                state.γ = max(min(sqrt(state.γ), log(10, state.γ + 1)), 1e-16)
            end
            if (ro > 0.05 && df > 0) || it == iter.itermax
                state.a1 = a1
                state.a2 = a2
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
                state.t = (Dates.now() - iter.t).value / 1e3
                return state, state
            end
            it += 1
        end
    else
        state.ψ += 1.0
        # univariate line search functions
        s = similar(state.∇f)
        s = -state.∇f
        x = state.x

        α, fx, lsa = OneDLineSearch(iter, state.∇f, state.fz, x, s)

        # summary
        x = y = state.z - α .* state.∇f
        gnorm = norm(state.∇f)
        state.a1 = α
        state.a2 = 0.0
        state.x = x
        state.y = y
        state.fx = fx
        state.ρ = 1.0
        state.dq = 0.0
        state.df = fz - fx
        state.d = x - z
        state.Δ = α * gnorm
        state.ϵ = gnorm
        state.it = it
        state.t = (Dates.now() - iter.t).value / 1e3
        return state, state
    end
end

drsom_stopping_criterion(tol, state::DRSOMState) =
    (abs.([state.a1 state.a2]) |> maximum <= 1e-20) || (state.ϵ <= tol) && abs(state.fz - state.fx) <= tol


function drsom_display(it, state::DRSOMState)
    if it == 1
        log = @sprintf("%5s | %5s | %10s | %8s | %8s | %7s | %7s | %7s | %7s | %6s | %2s | %6s \n",
            "k", "ψ", "f", "α1", "α2", "Δ", "|∇f|", "γ", "λ", "ρ", "kₜ", "t",
        )
        format_header(log)
        @printf("%s", log)
    end
    if mod(it, 30) == 0
        @printf("%5s | %5s | %10s | %8s | %8s | %7s | %7s | %7s | %6s | %7s | %2s | %6s \n",
            "k", "ψ", "f", "α1", "α2", "Δ", "|∇f|", "γ", "λ", "ρ", "kₜ", "t",
        )
    end
    @printf("%5d | %5d | %+.3e | %+.1e | %+.1e | %.1e | %.1e | %.1e | %.1e | %+.0e | %2d | %6.1f \n",
        it, state.ψ, state.fx, state.a1, state.a2, state.Δ, state.ϵ, state.γ, state.λ, state.ρ, state.it, state.t
    )
end

default_solution(::DRSOMIteration, state::DRSOMState) = state.x

"""
"""
# ReducedTrustRegion(;
#     maxit=10_000,
#     tol=1e-8,
#     stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
#     solution=default_solution,
#     verbose=false,
#     freq=100,
#     display=default_display,
#     kwargs...
# ) = IterativeAlgorithm(DRSOMIteration; maxit, stop, solution, verbose, freq, display, kwargs...)

# Aliases
