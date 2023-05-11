# beta: adaptive hsodm subproblem

# Base.@kwdef mutable struct AdaptiveHomogeneousSubproblem

using Roots
using SparseArrays
using ArnoldiMethod
using LDLFactorizations
try
    using MKL
catch e
    @warn("Unable to import MKL; fall back to BLAS")
end

Base.@kwdef mutable struct AR
    ALIAS::String = "AR"
    DESC::String = "AdaptiveRegularization"
    tol::Float64 = 1e-10
    itermax::Int64 = 20
    σ::Float64 = 1e3
    l::Float64 = 0.1
    u::Float64 = 0.1
    lₖ::Float64 = 0.1
    uₖ::Float64 = 0.1
    η₁::Float64 = 0.25
    η₂::Float64 = 0.95
    ρ::Float64 = 0.0
    γ₁::Float64 = 1.1
    γ₂::Float64 = 1.3
    γ₃::Float64 = 1.5
    # interval for σ
    lₛ::Float64 = 1
    uₛ::Float64 = 1e5
end

Base.@kwdef mutable struct ArnoldiInfo
    numops::Int
end



# adaptive GHMs wrapping ArC style
function AdaptiveHomogeneousSubproblem(B, iter, state, adaptive_param::AR; verbose::Bool=false)
    kᵥ = 1
    k₂ = 1
    homogeneous_eigenvalue.counter = 0
    _, λ₁, ξ, t₀, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)

    # non-adaptive mode
    if iter.adaptive === :none
        state.λ₁ = λ₁
        state.ξ = ξ
        return kᵥ, k₂, v, vn, vg, vHv
    end

    # adaptive mode
    h(t, θ) = sqrt(t^2 / (1 - t^2)) * max(θ, 0)
    # line search root-finding function
    function fa(δ, p)
        B[end, end] = δ
        _, λ₁, __, t₀, __unused__ = homogeneous_eigenvalue(B, iter, state)
        hv = log(h(t₀, -λ₁) + 1) - log(p + 1)
        return hv
    end

    # start iteration
    while true
        hv = h(t₀, -λ₁)
        fx₊ = iter.f(state.z + v)
        # model ratio
        adaptive_param.ρ = state.ρ = (fx₊ - state.fz) / (vg + vHv / 2 + hv / 6 * vn^3)
        # adaptive_param.ρ = state.ρ = -(fx₊ - state.fz) / vn^3
        bool_acc = (fx₊ < state.fz) && (adaptive_param.ρ >= adaptive_param.η₁)
        bool_adj = false
        if (kᵥ == 1) && verbose
            @printf("       |\n")
        end
        if verbose
            @printf("       |--- t:%+.0e, Δf:%+.0e, m:%+.0e, ρ:%+.0e\n",
                t₀, fx₊ - state.fz, vg + vHv / 2 + hv / 6 * vn^3, state.ρ
            )
            @printf("       |--- δ:%+.0e, θ:%.0e, |d|:%.0e, h:%.0e (σ:%+.0e)\n",
                state.δ, -λ₁, vn,
                hv, adaptive_param.σ
            )
        end
        if bool_acc
            state.λ₁ = λ₁
            state.ξ = ξ
        end
        if !bool_acc
            # you should increase regularization, 
            # by decrease delta
            # adaptive_param.σ = max(adaptive_param.σ, hv) * adaptive_param.γ₂
            if hv < adaptive_param.lₛ
                adaptive_param.σ = max(adaptive_param.σ, adaptive_param.lₛ) * adaptive_param.γ₂
            else
                adaptive_param.σ = max(adaptive_param.σ, hv) * adaptive_param.γ₂
            end
            bool_adj = true
            adaptive_param.l = state.δ - 1e4
            adaptive_param.u = state.δ
            adaptive_param.lₖ = adaptive_param.σ
            adaptive_param.uₖ = adaptive_param.σ * adaptive_param.γ₃

        elseif adaptive_param.ρ >= adaptive_param.η₂
            adaptive_param.σ = max(adaptive_param.σ, 1e1, hv) / adaptive_param.γ₁
            # adaptive_param.σ = adaptive_param.σ / adaptive_param.γ₁
            adaptive_param.l = state.δ - 1e4
            adaptive_param.u = state.δ + 1e4

            if abs(adaptive_param.σ) > 1e0
                bool_adj = true
            end
            adaptive_param.lₖ = 0 # max(adaptive_param.σ - 1e4, -1e1)
            adaptive_param.uₖ = adaptive_param.σ
        else
            # remain unchanged
            # `bool_adj = false`
        end

        if bool_adj
            # search for proper δ
            linesearch(state, fa, adaptive_param; verbose=verbose, tracker=tracker)
        end

        if bool_acc || kᵥ >= 10
            k₂ = homogeneous_eigenvalue.counter
            return kᵥ, k₂, v, vn, vg, vHv
        end
        B[end, end] = state.δ
        if verbose
            @printf("       |--- B:%s\n", B[end-1:end, end-1:end])
        end
        _, λ₁, ξ, t₀, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)

        kᵥ += 1
    end
end

# adaptive GHMs wrapping ArC style
function AdaptiveHomogeneousSubproblem(f::Function, iter, state, adaptive_param::AR; verbose::Bool=false)
    kᵥ = 1
    k₂ = 1
    homogeneous_eigenvalue.counter = 0
    _, λ₁, ξ, t₀, v, vn, vg, vHv = homogeneous_eigenvalue(f, iter, state)

    # non-adaptive mode
    if iter.adaptive === :none
        state.λ₁ = λ₁
        # printstyled(ξ)
        state.ξ = ξ[:, 1]
        return kᵥ, k₂, v, vn, vg, vHv
    end
    # todo, implement adaptive version.
end

function _inner_homogeneous_eigenvalue(
    B::Symmetric{Q,F}, iter, state
) where {Q<:Real,F<:Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}}}
    n = length(state.x)
    vals, vecs, info = eigenvalue(B, iter, state)
    λ₁ = vals |> real
    ξ = vecs[:, 1] |> real

    v = reshape(ξ[1:end-1], n)
    t₀ = abs(ξ[end])
    t₀ = min(max(t₀, 1e-4), 1 - 1e-4)

    # scale v if t₀ is big enough
    # reverse this v if g'v > 0
    v = v / t₀
    vg = (v'*state.∇f)[]
    bool_reverse_v = vg > 0
    v = (-1)^bool_reverse_v * v
    vg = (-1)^bool_reverse_v * vg
    vn = norm(v)
    vHv = v' * B[1:end-1, 1:end-1] * v

    return info.numops, λ₁, ξ, t₀, v, vn, vg, vHv
end

function _inner_homogeneous_eigenvalue(f::Function, iter, state)
    n = length(state.x)
    vals, vecs, info = eigenvalue(f, iter, state)

    λ₁ = vals[1]
    ξ = vecs[1]

    v = reshape(ξ[1:end-1], n)
    t₀ = abs(ξ[end])
    t₀ = min(max(t₀, 1e-4), 1 - 1e-4)

    # scale v if t₀ is big enough
    # reverse this v if g'v > 0
    v = v / t₀
    vg = (v'*state.∇f)[]
    bool_reverse_v = vg > 0
    v = (-1)^bool_reverse_v * v
    vg = (-1)^bool_reverse_v * vg
    vn = norm(v)
    vHv = v'state.∇fb

    return info.numops, λ₁, ξ, t₀, v, vn, vg, vHv
end



homogeneous_eigenvalue = Counting(_inner_homogeneous_eigenvalue)

function eigenvalue(
    B::Symmetric{Float64,F}, iter, state; bg=:arnoldi
) where {F<:Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}}}

    n = length(state.x)
    if bg == :krylov
        if iter.direction == :cold
            vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol, issymmetric=true, eager=true)
        else
            vals, vecs, info = KrylovKit.eigsolve(B, state.ξ, 1, :SR; tol=iter.eigtol, issymmetric=true, eager=true)
        end
        return vals[1], vecs[1], info
    end
    if bg == :arnoldi
        try
            # arnoldi is not stable?
            decomp, history = ArnoldiMethod.partialschur(B, nev=1, restarts=50000, tol=iter.eigtol, which=SR())
            vals, vecs = partialeigen(decomp)
            return vals[1], vecs, ArnoldiInfo(numops=1)
        catch
            vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol, issymmetric=true, eager=true)
            return vals[1], vecs[1], info
        end
    end
end


function eigenvalue(f::Function, iter, state; bg=:krylov)
    if bg == :krylov
        if iter.direction == :cold
            n = length(state.x)
            vals, vecs, info = KrylovKit.eigsolve(f, n + 1, 1, :SR, Float64; tol=iter.eigtol, issymmetric=true, eager=true)
        else
            vals, vecs, info = KrylovKit.eigsolve(f, state.ξ, 1, :SR; tol=iter.eigtol, issymmetric=true, eager=true)
        end
    end
    return vals, vecs, info
end


function linesearch(state, fa, adaptive_param; verbose::Bool=false, method::Symbol=:order2, tracker=tracker)
    # oracle
    atol = log((adaptive_param.uₖ - adaptive_param.lₖ) / 2 + 1)
    midpoint = log((adaptive_param.lₖ + adaptive_param.uₖ) / 2 + 1)

    if method == :bisection
        p = ZeroProblem(fa, (adaptive_param.l, adaptive_param.u))
        M = Roots.Bisection()
        if verbose
            @printf("       |--- δ:[%+.0e, %+.0e] => σ:[%+.0e, %+.0e]\n",
                adaptive_param.l, adaptive_param.u, adaptive_param.lₖ, adaptive_param.uₖ
            )
        end
    elseif method == :order2
        p = ZeroProblem(fa, state.δ)
        M = Roots.Order2B()
        if verbose
            @printf("       |--- δ₀: %+.0e => σ:[%+.0e, %+.0e]\n",
                state.δ, adaptive_param.lₖ, adaptive_param.uₖ
            )
        end
    end
    δ = solve(
        p, M;
        verbose=verbose,
        atol=atol,
        tracks=tracker,
        p=midpoint
    )
    ~isnan(δ) && (state.δ = δ)
    return δ
end



function NewtonStep(
    H::SparseMatrixCSC{R,T}, μ, g, state; verbose::Bool=false
) where {R<:Real,T<:Int}
    # d, __unused_info = KrylovKit.linsolve(
    #     H + μ * SparseArrays.I, -Vector(g);
    #     isposdef=true, issymmetric=true
    # )
    # d = -((H + μ * I) \ g)
    cc = ldl(H + μ * I)
    d = -cc \ g
    return 1, 1, d, norm(d), d' * state.∇f, d' * H * d
end

function NewtonStep(H::Matrix{R}, μ, g, state; verbose::Bool=false
) where {R<:Real}
    d = -((H + μ * I) \ g)
    return 1, 1, d, norm(d), d' * state.∇f, d' * H * d
end

function NewtonStep(f, g, state; verbose::Bool=false)
    d, __unused_info = KrylovKit.linsolve(
        f, -Vector(g);
        isposdef=true, issymmetric=true
    )
    return 1, 1, d, norm(d), d' * state.∇f, d' * f(d)
end

###############################################################################
# a vanilla bisection
###############################################################################

function bisection(h, adaptive_param, B, iter, state; verbose::Bool=false)
    throw(ErrorException("This is a vanilla version of bisection, use line-search instead"))
    l = adaptive_param.l
    u = adaptive_param.u
    if verbose
        @printf("       |--- δ:[%+.0e, %+.0e] => σ:[%+.0e, %+.0e]\n",
            l, u, adaptive_param.lₖ, adaptive_param.uₖ
        )
    end
    ls = 0
    k₂ = 0
    while true
        δ = (l + u) / 2
        B[end, end] = δ
        _, λ₁, ξ, t₀, __unused__ = homogeneous_eigenvalue(B, iter, state)

        hv = h(t₀, -λ₁)
        k₂ += 1
        ls += 1
        if hv > adaptive_param.uₖ
            # too small, increase 
            l = δ
        elseif hv < adaptive_param.lₖ
            u = δ
        else
            if verbose
                @printf("       |--- δ+:%+.0e, i:%+.0e, h(δ+):%+.0e\n",
                    δ, ls, hv
                )
            end
            state.δ = δ
            break
        end
        if abs(l - u) < 1e-2
            if verbose
                @printf("       |--- δ+:%+.0e, i:%+.0e, h(δ+):%+.0e\n",
                    δ, ls, hv
                )
                @warn("The Line-search on δ failed")
            end
            break
        end
    end
    return k₂
end
