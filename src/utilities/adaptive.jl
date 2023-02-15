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
    σ::Float64 = 0.1
    η₁::Float64 = 0.2
    η₂::Float64 = 0.8
    ρ::Float64 = 0.0
    γ₁::Float64 = 1.1 # successful
    γ₂::Float64 = 1.2 # very successful
    γ₃::Float64 = 1.5 # interval for σ
end



# arc style
function AdaptiveHomogeneousSubproblem(B, iter, state, adaptive_param::AR)
    kᵥ = 0
    k₂ = 1

    # justification function
    h(t, θ) = sqrt(t^2 / (1 - t^2)) * θ
    kᵥ, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)
    hv = h(state.ξ[end], -state.λ₁)

    if iter.adaptive === :none
        return true, kᵥ, k₂, v, vn, vg, vHv
    end

    fx₊ = iter.f(state.z + v)
    # model ratio
    adaptive_param.ρ = state.ρ = (fx₊ - state.fz) / (vg + vHv / 2 + hv / 6 * vn^3)
    bool_acc = (fx₊ < state.fz) && (adaptive_param.ρ >= adaptive_param.η₁)
    bool_adj = false
    if !bool_acc
        adaptive_param.σ *= adaptive_param.γ₂
        bool_adj = true
    elseif adaptive_param.ρ >= adaptive_param.η₂
        adaptive_param.σ /= adaptive_param.γ₁
        bool_adj = true
    else
        # remain unchanged
        # bool_adj = false
    end

    if bool_adj
        # search for proper δ
        l, u = -1e3, 1e3
        lₖ = adaptive_param.σ / adaptive_param.γ₃
        uₖ = adaptive_param.σ * adaptive_param.γ₃
        while true
            if hv > uₖ
                # too small, increase 
                l = state.δ
            elseif hv < lₖ
                u = state.δ
            else
                break
            end
            state.δ = (l + u) / 2
            B[end, end] = state.δ
            kᵥ, v, vn, vg, vHv = homogeneous_eigenvalue(B, iter, state)
            hv = h(state.ξ[end], -state.λ₁)
            k₂ += 1
        end
    end

    return bool_acc, kᵥ, k₂, v, vn, vg, vHv

end

function homogeneous_eigenvalue(B, iter, state)
    n = length(state.x)
    if iter.direction == :cold
        vals, vecs, info = KrylovKit.eigsolve(B, n + 1, 1, :SR, Float64; tol=iter.eigtol, eager=true)
    else
        try
            vals, vecs, info = KrylovKit.eigsolve(B, state.ξ, 1, :SR; tol=iter.eigtol, eager=true)
        catch
            printstyled(B)
        end
    end
    state.λ₁ = vals[1]
    state.ξ = ξ = vecs[1]
    kᵥ = info.numops
    v = reshape(ξ[1:end-1], n)
    t₀ = ξ[end]
    vg = (v'*state.∇f)[]
    bool_reverse_v = vg > 0
    # scale v if t₀ is big enough
    # reverse this v if g'v > 0
    (abs(t₀) > 1e-6) && (v = v / t₀)
    v = (-1)^bool_reverse_v * v
    vg = (-1)^bool_reverse_v * vg
    vn = norm(v)
    vHv = v' * B[1:end-1, 1:end-1] * v

    return kᵥ, v, vn, vg, vHv
end


