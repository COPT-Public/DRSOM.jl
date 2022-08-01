###############
# file: test_lp_smooth.jl
# project: src
# created Date: Tu Mar yyyy
# author: <<author>
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
###############


include("lp.jl")
include("helper.jl")

using ProximalOperators
using DRSOM
using ProximalAlgorithms
using Plots
using Printf
using LazyStack
using HTTP
using LaTeXStrings
using LinearAlgebra
using Statistics
using .LP

option_plot = false
########################################################
data = LP.load_diabetes_dataset()
name = "diabetes"

training_input = data[1:end-100, 1:end-1]
training_label = data[1:end-100, end]

test_input = data[end-99:end, 1:end-1]
test_label = data[end-99:end, end]

n_training, n_features = size(training_input)

input_loc = mean(training_input, dims=1) |> vec
input_scale = std(training_input, dims=1) |> vec

n, m = size(training_input)
onen = ones(n, 1)
x_hat = (training_input .- (onen * input_loc')) ./ (onen * input_scale')
x_A = [x_hat onen]
x_gram = x_A' * x_A
L = LinearAlgebra.opnorm(x_gram, 2)
λs = LinearAlgebra.eigvals(x_gram)

sort!(λs)
σ = max(0, λs[1])

########################################################
g = ProximalOperators.SqrNormL2(2)
training_loss(wb) = LP.mean_squared_error(
    training_label, LP.standardized_linear_model(wb, training_input, input_loc, input_scale)
)
f = LeastSquares(x_A, training_label, 1 / n)
f_composite(x) = training_loss(x) + LP.lpp(x, 2)

x0 = zeros(n_features + 1)
lowtol = 1e-6
tol = 1e-8
maxtime = 300
maxiter = 1e5
verbose = true
freq = 40

#######
iter_scale = 0
#######

method_objval = Dict{String,AbstractArray{Float64}}(
    "FISTA" => [],
    # "FISTA" => [],
    "SFISTA" => [],
    "RSOM" => [],
    "DR" => [],
    "DRLS" => []
)
method_state = Dict{String,Any}()

# fista
name, state, k, arr_obj = run_fista(copy(x0), f, g, L, σ)
method_objval[name] = copy(arr_obj)
iter_scale = max(iter_scale, k) # compute max plot scale
# drls
name, state, k, arr_obj = run_drls(copy(x0), f, g, L, σ)
method_objval[name] = copy(arr_obj)
iter_scale = max(iter_scale, k) # compute max plot scale
# rsom
name, state, k, arr_obj = run_drsomb(copy(x0), f_composite)
method_objval[name] = copy(arr_obj)
iter_scale = max(iter_scale, k) # compute max plot scale

method_objval_ragged = rstack([values(method_objval)...]; fill=NaN)


if option_plot
    pgfplotsx()

    pl_residual = plot(
        1:iter_scale,
        method_objval_ragged,
        label=permutedims([keys(method_objval)...]),
        xscale=:log2,
        yscale=:log10,
        xlabel="Iteration", ylabel=L"Objective: $f$",
        title=L"\textrm{Least Squares with }  \mathcal{L}_p \textrm{ Regularization}: 
        \frac{1}{2}\|Ax-b\|^2 + \|x\|_p^p, p = 2",
        size=(1080, 720),
        # yticks=[1e-2, 1, 100, 1000, 1000],
        # xticks=[1, 10, 100, 200, 500, 1000, 10000],
        dpi=500
    )

    name = @sprintf("rsom_lp_smooth_p-%d", 2)
    savefig(pl_residual, @sprintf("/tmp/%s.tikz", name))
    savefig(pl_residual, @sprintf("/tmp/%s.tex", name))
    savefig(pl_residual, @sprintf("/tmp/%s.pdf", name))
    savefig(pl_residual, @sprintf("/tmp/%s.png", name))
end