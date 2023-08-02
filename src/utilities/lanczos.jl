# A lanczos implementation, modified from,
#    https://github.com/JuliaInv/KrylovMethods.jl.git
using SparseArrays
using LinearAlgebra

function LanczosTridiag(A::SparseMatrixCSC{T1,Int}, b::Array{T2,1}; kwargs...) where {T1,T2}
    x = zeros(promote_type(T1, T2), size(A, 2)) # pre-allocate
    return LanczosTridiag(v -> mul!(x, transpose(A), v, 1.0, 0.0), b; kwargs...) # multiply with transpose of A for efficiency
end


LanczosTridiag(A, b; kwargs...) = LanczosTridiag(x -> A * x, b; kwargs...)
LanczosTrustRegionBisect(A, b, Δ; kwargs...) = LanczosTrustRegionBisect(x -> A * x, b, Δ; kwargs...)


function LanczosTridiag(
    A::Function,
    b::Vector;
    tol=1e-5,
    bool_reorth::Bool=false,
    k::Int=(b |> length)
)

    n = length(b)
    x = zeros(n)

    # pre-allocate space for tri-diagonalization and basis
    γ = zeros(k)
    α = zeros(k)
    V = zeros(n, k)


    γ[1] = norm(b)
    V[:, 1] = copy(b) / γ[1]
    u = A(V[:, 1])

    j = 1
    while j <= k - 1
        α[j] = dot(V[:, j], u)
        u = u - α[j] * V[:, j]
        if bool_reorth # full re-orthogonalization
            for i = 1:j
                u -= V[:, i] * dot(V[:, i], u)
            end
        end
        γⱼ = norm(u)
        V[:, j+1] = u / γⱼ
        γ[j+1] = γⱼ
        if γ[j+1] < tol
            break
        end
        u = A(V[:, j+1]) - γ[j+1] * V[:, j]
        j += 1
    end
    j = min(j, k - 1)
    T = SymTridiagonal(
        α[1:j], γ[2:j]
    )
    return T, V[:, 1:j], γ[1:j+1], j
end

function LanczosTrustRegionBisect(
    A::Function,
    b::Vector,
    Δ::Real;
    tol=1e-5,
    bool_reorth::Bool=false,
    k::Int=(b |> length),
    ϵ=1e-4,
    ϵₗ=1e-2
)
    # do tridiagonalization
    T, V, γ, kᵢ = LanczosTridiag(A, b; tol=1e-5, bool_reorth=true, k=k)

    # a mild estimate
    # λₗ = maximum(sum(abs, T, dims=1)[:] - diag(T))
    λₗ = max(-eigvals(T, 1:1)[], 0)
    λᵤ = (b |> norm) / Δ
    bₜ = V' * b
    λₖ = λₗ
    Sₖ = ldlt(T + λₖ * I)
    dₖ = Sₖ \ bₜ
    kₗ = 1
    if (dₖ |> norm) <= Δ
        return V * dₖ
    end
    # otherwise, we initialize a search procedure
    # using bisection
    while abs(λᵤ - λₗ) > ϵₗ
        λₖ = (λₗ + λᵤ) / 2
        Sₖ = ldlt(T + λₖ * I)
        dₖ = Sₖ \ bₜ
        dₙ = dₖ |> norm
        kₗ += 1
        if dₙ <= Δ - ϵ
            λᵤ = λₖ
        elseif dₙ >= Δ + ϵ
            λₗ = λₖ
        else
            break
        end
    end
    return V * dₖ, λₖ, kₗ
end

function LanczosTrustRegionBisect(
    T::SymTridiagonal{F},
    V::Matrix,
    b::Vector,
    Δ::Real,
    λₗ::Real,
    λᵤ::Real;
    tol=1e-5,
    bool_reorth::Bool=false,
    k::Int=(b |> length),
    ϵ=1e-4,
    ϵₗ=1e-2
) where {F}
    # a mild estimate
    bₜ = V' * b
    λₖ = λₗ
    Sₖ = ldlt(T + λₖ * I)
    dₖ = Sₖ \ bₜ
    kₗ = 1
    if (dₖ |> norm) <= Δ
        return V * dₖ, λₖ, kₗ
    end
    # otherwise, we initialize a search procedure
    # using bisection
    while abs(λᵤ - λₗ) > ϵₗ
        λₖ = (λₗ + λᵤ) / 2
        Sₖ = ldlt(T + λₖ * I)
        dₖ = Sₖ \ bₜ
        dₙ = dₖ |> norm
        kₗ += 1
        if dₙ <= Δ - ϵ
            λᵤ = λₖ
        elseif dₙ >= Δ + ϵ
            λₗ = λₖ
        else
            break
        end
    end
    return V * dₖ, λₖ, kₗ
end

function lanczos_solve_tr_inexact(
    A::Function,
    b::Vector,
    Δ::Real;
    tol=1e-5,
    bool_reorth::Bool=false,
    k::Int=(b |> length)
)
end