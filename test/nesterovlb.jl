using LinearAlgebra, SparseArrays
using ForwardDiff, ADNLPModels


Base.@kwdef mutable struct NesterovHardCase
    n::Int
    k::Int
    p::Int
    A::SparseMatrixCSC{Float64,Int}
    f::Function
    g::Function
    hvp::Function
    H::Function
    η::Function
    ∇η::Function
    Hη::Function
    x₀::Vector{Float64}
    xₛ::Vector{Float64}
    fₛ::Float64

    function NesterovHardCase(n::Int, k::Int, p::Int)
        this = new()
        this.n, this.k, this.p = n, k, p
        this.f, this.g, this.hvp, this.H, this.η, this.∇η, this.Hη, this.A = nesterovlb(n, k, p)
        this.x₀ = zeros(n)
        this.xₛ = max.(0.0, [k - i + 1 for i in 1:n])
        this.fₛ = -k * p / (p + 1)
        return this
    end

end

function check_nesterovlb(data::NesterovHardCase, x::Vector{Float64})
    @info "check result" norm(x - data.xₛ) norm(data.g(x)) abs(data.f(x) - data.fₛ)
end

function get_nlp_nesterovlb(data::NesterovHardCase)
    return ADNLPModel(data.f, data.x₀)
end



@doc raw"""
    nesterovlb(n::Int)

Return the n-dimensional lower bound function of Nesterov.
"""
function nesterovlb(n::Int, k::Int, p::Int)
    U = spdiagm(0 => ones(k), 1 => -ones(k - 1))
    A = blockdiag(U, spdiagm(0 => ones(n - k)))
    η(x) = 1 / (p + 1) * sum((x .|> abs) .^ (p + 1))
    f(x) = η(A * x) - x[1]

    ∇η(x) = abs.(x) .^ p .* sign.(x)
    Hη(x) = spdiagm(p .* abs.(x) .^ (p - 1))

    function g(x)
        z = A' * ∇η(A * x)
        z[1] -= 1
        return z
    end
    function hvp(w, v, Hv; eps=1e-5)
        gn = g(w + eps * v)
        gf = g(w)
        copyto!(Hv, (gn - gf) / eps)
    end

    H(x) = A' * Hη(A * x) * A

    return f, g, hvp, H, η, ∇η, Hη, A
end
