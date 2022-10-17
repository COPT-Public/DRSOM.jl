
module LP
using ArgParse
using HTTP
using LinearAlgebra
using ProximalOperators
using Statistics
using Distributions

splitlines(s) = split(s, "\n")
splitfields(s) = split(s, "\t")
parsefloat64(s) = parse(Float64, s)

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

function load_diabetes_dataset()
    res = HTTP.request("GET", "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt")
    lines = res.body |> String |> strip |> splitlines
    return hcat((line |> splitfields .|> parsefloat64 for line in lines[2:end])...)'
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
    nnz::Float64 = 0.5
    λ::Float64 = 1 / 2
end

function create_random_lp(params::LPMinimizationParams)
    n = params.n
    m = params.m
    D = Normal(0.0, 1.0)
    A = rand(D, (n, m)) .* rand(Bernoulli(0.15), (n, m))
    v = rand(Normal(0.0, 1 / n), m) .* rand(Bernoulli(0.5), m)
    b = A * v + rand(D, (n))
    return A, v, b
end

function huberloss(λ, δ, x::Real)
    if abs(x) <= δ
        return 0.5 * λ / δ * x^2
    else
        return λ * (abs(x) - δ / 2)
    end
end

function huberloss(λ, δ, x::Vector)
    cc = huberloss.(λ, δ, x)
    return cc |> sum
end
# a Lipschitz smoothed version of |x|_p^p, 
# see: 
# [1] Ge, D., Jiang, X., Ye, Y.: A note on the complexity of Lp minimization. Mathematical Programming. 129, 285–299 (2011). 
# [2] Chen, X., Ge, D., Wang, Z., Ye, Y.: Complexity of unconstrained $$L_2-L_p$$ minimization. Math. Program. 143, 371–383 (2014). https://doi.org/10.1007/s10107-012-0613-0
# using function (|x| + ϵ)^p
function smoothlp(λ, ϵ, p::Real, x::Real)
    return λ * (abs(x) + ϵ)^p

end

function smoothlp(λ, ϵ, p::Real, x::Vector)
    smoothlp.(λ, ϵ, p, x) |> sum
end

function smoothlpg(λ, ϵ, p::Real, x::Real)
    λ * p * (abs(x) + ϵ)^(p - 1) * sign(x)
end

function smoothlpg(λ, ϵ, p::Real, x::Vector)
    smoothlpg.(λ, ϵ, p, x)
end

function smoothlph(λ, ϵ, p::Real, x::Real)
    λ * p * (p - 1) * (abs(x) + ϵ)^(p - 2)
end

function smoothlph(λ, ϵ, p::Real, x::Vector)
    smoothlph.(λ, ϵ, p, x) |> Diagonal
end


# a really smoothed version
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

end # module