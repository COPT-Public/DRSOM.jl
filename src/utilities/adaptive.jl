# beta: adaptive hsodm subproblem

# Base.@kwdef mutable struct AdaptiveHomogeneousSubproblem

adaptive_rule_angle(vg, gnorm, vn) = (vg |> abs) < (min(gnorm, 1) * vn / 5)

function adaptive(vg::Float64, gnorm::Float64, vn::Float64, mode; kwds...)
    if mode === :none
        return false # always accept
    elseif mode === :angle
        return adaptive_rule_angle(vg, gnorm, vn)
    end
    return true
end

# style 1
function adaptive_subproblem(B, iter, state)
    kᵥ = 0
    k₂ = 0
    gnorm = norm(state.∇f)
    while true
        k₂ += 1
        kᵥ, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)
        cont = adaptive(vg, gnorm, vn, iter.adaptive)
        if cont && (k₂ < 50)
            δₖ -= sqrt(gnorm) / 40
            B[end, end] = δₖ
        else
            return kᵥ, k₂, v, vn, vg, vHv
        end
    end
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
    γ₃::Float64 = 2
    # interval for σ
    lₛ::Float64 = 1
    uₛ::Float64 = 1e5
end



# arc style
function AdaptiveHomogeneousSubproblem(B, iter, state, adaptive_param::AR; EXTRA_VERBOSE::Bool=false)
    kᵥ = 1
    k₂ = 1

    _, λ₁, ξ, t₀, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)


    if iter.adaptive === :none
        state.λ₁ = λ₁
        state.ξ = ξ
        return kᵥ, k₂, v, vn, vg, vHv
    end

    h(t, θ) = sqrt(t^2 / (1 - t^2)) * max(θ, 0)

    while true
        hv = h(t₀, -λ₁)
        fx₊ = iter.f(state.z + v)
        # model ratio
        adaptive_param.ρ = state.ρ = (fx₊ - state.fz) / (vg + vHv / 2 + hv / 6 * vn^3)
        # adaptive_param.ρ = state.ρ = -(fx₊ - state.fz) / vn^3
        bool_acc = (fx₊ < state.fz) && (adaptive_param.ρ >= adaptive_param.η₁)
        bool_adj = false
        if (kᵥ == 1) && EXTRA_VERBOSE
            @printf("       |\n")
        end
        if EXTRA_VERBOSE
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
            adaptive_param.lₖ = max(adaptive_param.σ, hv)
            adaptive_param.uₖ = adaptive_param.σ * 2

        elseif adaptive_param.ρ >= adaptive_param.η₂
            adaptive_param.σ = max(adaptive_param.σ, 1e1, hv) / adaptive_param.γ₁
            # adaptive_param.σ = adaptive_param.σ / adaptive_param.γ₁
            adaptive_param.l = state.δ - 1e4
            adaptive_param.u = state.δ + 1e4
            println(adaptive_param.l, adaptive_param.u)
            if abs(adaptive_param.u - adaptive_param.l) > 1e1
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
            k₂ += linesearch(h, adaptive_param, B, iter, state; EXTRA_VERBOSE=EXTRA_VERBOSE)
        end

        if bool_acc || kᵥ >= 20
            return kᵥ, k₂, v, vn, vg, vHv
        end
        B[end, end] = state.δ
        if EXTRA_VERBOSE
            @printf("       |--- B:%s\n", B[end-1:end, end-1:end])
        end
        _, λ₁, ξ, t₀, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)
        k₂ += 1
        kᵥ += 1
    end
end

function homogeneous_eigenvalue(B, iter, state)
    n = length(state.x)
    vals, vecs, info = eigenvalue(B, iter, state)
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
    vHv = v' * B[1:end-1, 1:end-1] * v

    return info.numops, λ₁, ξ, t₀, v, vn, vg, vHv
end


function eigenvalue(B, iter, state)
    if iter.direction == :cold
        vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol, eager=true)
    else
        vals, vecs, info = KrylovKit.eigsolve(B, state.ξ, 1, :SR; tol=iter.eigtol, eager=true)
    end
    return vals, vecs, info
end

function linesearch(h, adaptive_param, B, iter, state; EXTRA_VERBOSE::Bool=false)
    l = adaptive_param.l
    u = adaptive_param.u
    if EXTRA_VERBOSE
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
            if EXTRA_VERBOSE
                @printf("       |--- δ+:%+.0e, i:%+.0e, h(δ+):%+.0e\n",
                    δ, ls, hv
                )
            end
            state.δ = δ
            break
        end
        if abs(l - u) < 1e-2
            if EXTRA_VERBOSE
                @printf("       |--- δ+:%+.0e, i:%+.0e, h(δ+):%+.0e\n",
                    δ, ls, hv
                )
            end
            @warn("The Line-search on δ failed")
            break
        end
    end
    return k₂
end