# A lanczos implementation, modified from,
#    https://github.com/JuliaInv/KrylovMethods.jl.git
using SparseArrays
using LinearAlgebra

function LanczosTridiag(A::SparseMatrixCSC{T1,Int}, b::Array{T2,1}; kwargs...) where {T1,T2}
    x = zeros(promote_type(T1, T2), size(A, 2)) # pre-allocate
    return LanczosTridiag(v -> mul!(x, transpose(A), v, 1.0, 0.0), b; kwargs...) # multiply with transpose of A for efficiency
end


LanczosTridiag(A, b; kwargs...) = LanczosTridiag(x -> A * x, b; kwargs...)
LanczosTridiag(A, b, lc; kwargs...) = LanczosTridiag(x -> A * x, b, lc; kwargs...)
LanczosTrustRegionBisect(A, b, Δ; kwargs...) = LanczosTrustRegionBisect(x -> A * x, b, Δ; kwargs...)
InexactLanczosTrustRegionBisect(A, b, Δ, lc; kwargs...) = InexactLanczosTrustRegionBisect(x -> A * x, b, Δ, lc; kwargs...)

mutable struct Lanczos
    n::Int
    k::Int # max  dim
    j::Int # used dim
    α::Vector{Float64}
    γ::Vector{Float64}
    V::Matrix{Float64}
    Lanczos(n::Int, k::Int, b) = begin
        α::Vector{Float64} = zeros(Float64, k) # diag
        γ::Vector{Float64} = zeros(Float64, k - 1) # offdiag
        V = zeros(n, k)
        j = 1
        new(n, k, j, α, γ, V)
    end
end

Base.@kwdef mutable struct LanczosInfo
    λ₁::Float64 = 0.0
    kₗ::Float64 = 0.0
    εₙ::Float64 = 0.0
end

function LanczosTridiag(
    A::Function,
    b::Vector;
    tol=1e-5,
    k::Int=(b |> length),
    bool_reorth::Bool=false
)

    n = length(b)

    # pre-allocate space for tri-diagonalization and basis
    γ = zeros(k - 1)
    α = zeros(k)
    V = zeros(n, k)

    V[:, 1] = copy(b) / norm(b)
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
        γ[j] = γⱼ
        if γ[j] < tol
            break
        end
        u = A(V[:, j+1]) - γ[j] * V[:, j]
        j += 1
    end
    j = min(j, k - 1)
    T = SymTridiagonal(
        α[1:j], γ[1:j-1]
    )
    return T, V[:, 1:j], γ[1:j-1], j
end

function LanczosTridiag(
    A::Function,
    b::Vector,
    lc::Lanczos;
    tol=1e-5,
    k::Int=(b |> length),
    bool_reorth::Bool=false
)
    j = lc.j
    while j <= k - 1
        @info j
        if j == 1
            lc.V[:, 1] = copy(b) / norm(b)
            u = A(lc.V[:, j])
        else
            u = A(lc.V[:, j]) - lc.γ[j-1] * lc.V[:, j-1]
        end
        lc.α[j] = dot(lc.V[:, j], u)
        u = u - lc.α[j] * lc.V[:, j]
        if bool_reorth # full re-orthogonalization
            for i = 1:j
                u -= lc.V[:, i] * dot(lc.V[:, i], u)
            end
        end
        γⱼ = norm(u)
        lc.V[:, j+1] = u / γⱼ
        lc.γ[j] = γⱼ

        # use j below
        if lc.γ[j] < tol
            break
        end
        j += 1
    end
    lc.j = j
    T = SymTridiagonal(
        view(lc.α, 1:j), view(lc.γ, 1:j-1)
    )
    V = view(lc.V, :, 1:j)
    return T, V[:, 1:j], lc.γ[1:j-1], j
end

function LanczosTrustRegionBisect(
    A::Function,
    b::Vector,
    Δ::Real;
    tol=1e-5,
    k::Int=(b |> length),
    ϵ=1e-4,
    ϵₗ=1e-4,
    bool_reorth::Bool=false,
    bool_interior::Bool=false
)
    # do tridiagonalization
    T, V, γ, kᵢ = LanczosTridiag(A, b; tol=tol, bool_reorth=bool_reorth, k=k)
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

    while true
        Sₖ = ldlt(T + λᵤ * I)
        dₖ = Sₖ \ bₜ
        dₙ = dₖ |> norm
        kₗ += 1
        if dₙ <= Δ
            break
        else
            λᵤ *= 2
        end
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
            bool_interior && break
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
    ϵ=1e-4,
    ϵₗ=1e-2,
    bool_interior::Bool=false
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
    while true
        Sₖ = ldlt(T + λᵤ * I)
        dₖ = Sₖ \ bₜ
        dₙ = dₖ |> norm
        kₗ += 1
        if dₙ <= Δ
            break
        else
            λᵤ *= 2
        end
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
            bool_interior && break
            λᵤ = λₖ
        elseif dₙ >= Δ + ϵ
            λₗ = λₖ
        else
            break
        end
    end
    return V * dₖ, λₖ, kₗ
end

@doc """
Ξ - constants in the inexactness assumption
"""
function InexactLanczosTrustRegionBisect(
    A::Function,
    b::Vector,
    Δ::Real,
    lc::Lanczos;
    tol=1e-5,
    σ=0.0,
    k::Int=(b |> length),
    ϵ=1e-4,
    ϵₗ=1e-2,
    Ξ=1e-3,
    bool_reorth::Bool=false,
    bool_interior::Bool=false
)
    kₗ = 1
    xₖ = zeros(lc.n)
    λₖ = 0.0
    j = lc.j


    while true
        if j <= k - 1 # degree not desired; expand
            if j == 1
                lc.V[:, 1] = copy(b) / norm(b)
                u = A(lc.V[:, j])
            else
                u = A(lc.V[:, j]) - lc.γ[j-1] * lc.V[:, j-1]
            end
            lc.α[j] = dot(lc.V[:, j], u)
            u = u - lc.α[j] * lc.V[:, j]
            if bool_reorth # full re-orthogonalization
                for i = 1:j
                    u -= lc.V[:, i] * dot(lc.V[:, i], u)
                end
            end
            γⱼ = norm(u)
            lc.V[:, j+1] = u / γⱼ
            lc.γ[j] = γⱼ

            j += 1
            lc.j = j
        end

        # now compute solution
        T = SymTridiagonal(
            view(lc.α, 1:j), view(lc.γ, 1:j-1)
        ) + σ * I
        V = view(lc.V, :, 1:j)

        # a mild estimate
        # minimum eigenvalue
        λ₁ = eigvals(T, 1:1)[]
        # λₗ = max(-λ₁, λₖ)
        λₗ = max(-λ₁, 0)
        λᵤ = (b |> norm) / Δ + λₗ
        bₜ = V' * b
        λₖ = λₗ
        Sₖ = ldlt(T + λₖ * I)
        dₖ = Sₖ \ bₜ
        dₙ = (dₖ |> norm)

        if dₙ <= Δ
            # check if minimum λ produces an interior point
            xₖ = V * dₖ
        else
            while true
                Sₖ = ldlt(T + λᵤ * I)
                dₖ = Sₖ \ bₜ
                dₙ = dₖ |> norm
                kₗ += 1
                if dₙ <= Δ
                    break
                else
                    λᵤ *= 2
                end
            end
            @debug begin
                """
                [λₗ, λᵤ] [$λₗ,$λᵤ,$((b |> norm) / Δ + max(-λ₁, 0))]

                """
            end
            # otherwise, we initialize a search procedure
            # using bisection (sway to λₗ)
            while abs(λᵤ - λₗ) > ϵₗ
                λₖ = λₗ * 0.5 + λᵤ * 0.5
                Sₖ = ldlt(T + λₖ * I)
                dₖ = Sₖ \ bₜ
                dₙ = dₖ |> norm
                kₗ += 1
                @debug begin
                    """
                     $dₙ $λₖ $λₗ $λᵤ
                    """
                end
                if dₙ <= Δ - ϵ
                    bool_interior && break
                    λᵤ = λₖ
                elseif dₙ >= Δ + ϵ
                    λₗ = λₖ
                else
                    break
                end
            end
            xₖ = V * dₖ
        end
        εₙ = (A(xₖ) + (σ + λₖ) * xₖ - b) |> norm
        @debug begin
            """
            Κ    $j
            abs  $εₙ, $(dₙ^2)
            ls   $kₗ 
             λₖ  $λₖ
            |d|  $dₙ : $Δ
            [λₗ, λᵤ] [$λₗ,$λᵤ]

            """
        end
        if (dₙ^2 * Ξ >= εₙ) || (lc.γ[j-1] < tol) || j >= k
            return xₖ, λₖ, LanczosInfo(εₙ=εₙ, kₗ=kₗ, λ₁=λ₁)
        end

    end
    return xₖ, λₖ, LanczosInfo(εₙ=εₙ, kₗ=kₗ, λ₁=λ₁)
end