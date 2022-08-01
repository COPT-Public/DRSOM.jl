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
using LineSearches
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
λ = 10

########################################################
training_loss(wb) = LP.mean_squared_error(
        training_label, LP.standardized_linear_model(wb, training_input, input_loc, input_scale)
)
f_composite(x) = training_loss(x)
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

name, state, k, arr_obj = run_drsomb(copy(x0), f_composite; maxiter=maxiter)
res1 = Optim.optimize(f_composite, x0, GradientDescent(;
                alphaguess=LineSearches.InitialHagerZhang(),
                linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
res2 = Optim.optimize(f_composite, x0, LBFGS(;
                linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)