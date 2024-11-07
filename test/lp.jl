@doc raw"""
file: lp.jl
author: Chuwen Zhang
-----
last Modified: Sat Dec 31 2022
modified By: Chuwen Zhang
-----
(c) 2022 Chuwen Zhang
-----
Module for testing Lp sparse minimization problems
see the papers:
1. Chen, X.: Smoothing methods for nonsmooth, nonconvex minimization. Math. Program. 134, 71–99 (2012). https://doi.org/10.1007/s10107-012-0569-0
2. Chen, X., Ge, D., Wang, Z., Ye, Y.: Complexity of unconstrained $L_2-L_p$ minimization. Math. Program. 143, 371–383 (2014). https://doi.org/10.1007/s10107-012-0613-0
3. Ge, D., Jiang, X., Ye, Y.: A note on the complexity of Lp minimization. Mathematical Programming. 129, 285–299 (2011). https://doi.org/10.1007/s10107-011-0470-2
"""
module LP
using ArgParse
using LinearAlgebra
using ProximalOperators
using Printf
using SparseArrays

splitlines(s) = split(s, "\n")
splitfields(s) = split(s, "\t")
parsefloat64(s) = parse(Float64, s)

push_optim_result_to_string(size, res, name) = string(
    size,
    ",$(name),",
    @sprintf("%d,%.4e,%.4e,%.3f\n",
        res.trace[end].iteration,
        res.trace[end].value,
        res.trace[end].g_norm,
        res.time_run
    )
)
push_state_to_string(size, state, name) = string(
    size,
    ",$(name),",
    @sprintf("%d,%.4e,%.4e,%.3f\n",
        k,
        state.fx,
        norm(state.∇f),
        state.t
    )
)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n"
        help = "number of samples"
        arg_type = Int
        default = 20
        "--m"
        help = "number of features"
        arg_type = Int
        default = 100
        "--p"
        help = "choice of p norm"
        arg_type = Float64
        default = 1.0
        "--nnz"
        help = "sparsity"
        arg_type = Float64
        default = 0.5
    end
    _args = parse_args(s, as_symbols=true)
    return LP.LPMinimizationParams(; _args...)
end

linear_model(wb, input) = input * wb[1:end-1] .+ wb[end]

function standardized_linear_model(wb, input, input_loc, input_scale)
    w_scaled = wb[1:end-1] ./ input_scale
    wb_scaled = vcat(w_scaled, wb[end] - dot(w_scaled, input_loc))
    return linear_model(wb_scaled, input)
end


mean_squared_error(label, output) = mean((output .- label) .^ 2) / 2

# define useful regularization
# lp(x) = |x|_p^p
lp(x, p) = LinearAlgebra.norm(x, p)
lpp(x, p) = lp(x, p)^p
l4(x) = lpp(x, 4)
l3(x) = lpp(x, 3)

Base.@kwdef mutable struct LPMinimizationParams
    n::Int64 = 10
    m::Int64 = 100
    p::Float64 = 1
    nnz::Float64 = 0.15
    nnzv::Float64 = 0.5
    λ::Float64 = 5e-2
end

function create_random_lp(params::LPMinimizationParams)
    n = params.n
    m = params.m
    A = sprandn(n, m, params.nnz)
    v = sprandn(m, 1, params.nnzv)
    v = Matrix(v)[:]
    b = A * v #+ randn(n)
    return A, v, b
end

# a Lipschitz smoothed version of |x|_p^p, 
# see: 
# [1] Ge, D., Jiang, X., Ye, Y.: A note on the complexity of Lp minimization. Mathematical Programming. 129, 285–299 (2011). 
# [2] Chen, X., Ge, D., Wang, Z., Ye, Y.: Complexity of unconstrained $$L_2-L_p$$ minimization. Math. Program. 143, 371–383 (2014).
function huberlike(λ, ϵ, p::Real, x::Real)
    if abs(x) > ϵ
        return λ * abs(x)^p
    else
        return λ * (ϵ / 2 + x^2 / ϵ / 2)^p
    end
end

function huberlike(λ, ϵ, p::Real, x::Vector)
    huberlike.(λ, ϵ, p, x) |> sum
end

function huberlike(λ, ϵ, p::Real, x)
    huberlike.(λ, ϵ, p, x) |> sum
end

function huberlikeg(λ, ϵ, p::Real, x::Real)
    if abs(x) > ϵ
        λ * p * (abs(x))^(p - 1) * sign(x)
    else
        λ * p * x / ϵ * (ϵ / 2 + x^2 / ϵ / 2)^(p - 1)
    end
end

function huberlikeg(λ, ϵ, p::Real, x::Vector)
    huberlikeg.(λ, ϵ, p, x)
end

function huberlikeh(λ, ϵ, p::Real, x::Real)
    if abs(x) > ϵ
        λ * p * (p - 1) * (abs(x))^(p - 2)
    else
        λ * p / ϵ * (ϵ / 2 + x^2 / ϵ / 2)^(p - 1) + λ * p * (x / ϵ)^2 * (p - 1) * (ϵ / 2 + x^2 / ϵ / 2)^(p - 2)
    end
end

function huberlikeh(λ, ϵ, p::Real, x::Vector)
    huberlikeh.(λ, ϵ, p, x) |> Diagonal
end

function vec_of_sparsevec_to_sparsematrix(vec::Vector{SparseVector{Float64,Int}})
    ncols = length(X)
    # Determine the number of rows
    nrows = maximum(v.n for v in X)

    # Calculate total number of non-zero elements
    total_nnz = sum(length(v.indices) for v in X)

    # Preallocate column pointers (ncols + 1)
    colptr = Vector{Int}(undef, ncols + 1)
    colptr[1] = 1  # Start at index 1

    for j in 1:ncols
        colptr[j+1] = colptr[j] + length(X[j].indices)
    end

    # Preallocate row indices and values
    row_indices = Vector{Int}(undef, total_nnz)
    values = Vector{Float64}(undef, total_nnz)

    # Fill row_indices and values
    ptr = 1
    for j in 1:ncols
        sv = X[j]
        for k in 1:length(sv.indices)
            row_indices[ptr] = sv.indices[k]
            values[ptr] = sv.values[k]
            ptr += 1
        end
    end

    # Construct the SparseMatrixCSC
    S = SparseMatrixCSC(nrows, ncols, colptr, row_indices, values)
    return S
end


end # module