###############
# project: RSOM
# created Date: Tu Mar 2022
# author: <<author>
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test RSOM on smoothed L2-Lp minimization problems,
# Comparison of RSOM and A "real" second-order mothod (Newton-trust-region)
# For L2-Lp minimization, see the paper by X. Chen
# 1. Chen, X.: Smoothing methods for nonsmooth, nonconvex minimization. Math. Program. 134, 71–99 (2012). https://doi.org/10.1007/s10107-012-0569-0
# 2. Chen, X., Ge, D., Wang, Z., Ye, Y.: Complexity of unconstrained $$L_2-L_p$$ minimization. Math. Program. 143, 371–383 (2014). https://doi.org/10.1007/s10107-012-0613-0
# 3. Ge, D., Jiang, X., Ye, Y.: A note on the complexity of Lp minimization. Mathematical Programming. 129, 285–299 (2011). https://doi.org/10.1007/s10107-011-0470-2
###############

include("helper.jl")
include("lp.jl")

using ProximalOperators
using LineSearches
using Optim
using DRSOM
using ProximalAlgorithms
using Random
using Distributions
using Plots
using Printf
using LazyStack
using KrylovKit
using HTTP
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using ArgParse
using Optim
using .LP


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n"
        help = "number of samples"
        arg_type = Int
        default = 100
        "--m"
        help = "number of features"
        arg_type = Int
        default = 50
        "--p"
        help = "choice of p norm"
        arg_type = Float64
        default = 3.0
    end
    _args = parse_args(s, as_symbols=true)
    return LP.LPMinimizationParams(; _args...)
end
Random.seed!(2)
params = parse_commandline()
n = params.n
m = params.m

# create a sparse instance
D = Uniform(0.0, 1.0)
A = rand(D, (n, m)) .* rand(Bernoulli(0.65), (n, m))
xt = rand(D, m) .* rand(Bernoulli(0.65), m)
b = A * xt

x0 = zeros(params.m)

# duplicate

gram = A' * A

# find Lipschitz constant
L, _ = LinearOperators.normest(gram, 1e-4)
λ = LinearAlgebra.normInf(A'b) / 10
# find convexity constant
# vals, vecs, info = KrylovKit.eigsolve(A'A, 1, :SR)
σ = 0.0
@printf("preprocessing finished\n")

########################################################
f = ProximalOperators.LeastSquares(A, b)
# direct evaluation
# smooth 1
# f_composite(x) = 1 / 2 * (A * x - b)' * (A * x - b) + LP.smoothlp(λ, 0.1, params.p, x)
# g(x) = A' * (A * x - b) + LP.smoothlpg(λ, 0.1, params.p, x)
# H(x) = A' * A + LP.smoothlph(λ, 0.1, params.p, x)

# huberlike
f_composite(x) = 1 / 2 * LP.l4(A * x - b) + LP.huberlike(λ, 1e-1, params.p, x)
# g(x) = A' * (A * x - b) + LP.smoothlpg(λ, 0.1, params.p, x)
# H(x) = A' * A + LP.smoothlph(λ, 0.1, params.p, x)

lowtol = 1e-6
tol = 1e-6
maxtime = 300
maxiter = 1e5
verbose = true
freq = 40

#######
iter_scale = 0
#######

method_objval = Dict{String,AbstractArray{Float64}}()
method_state = Dict{String,Any}()

name, state, k, arr_obj = run_drsomb(copy(x0), f_composite)
# name, state, k, arr_obj = run_rsomd(copy(x0), f_composite, g, H; tol=1e-7)

# compare with GD and LBFGS, Trust region newton,
options = Optim.Options(
    g_tol=1e-6,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=50,
)
res1 = Optim.optimize(f_composite, x0, GradientDescent(;
        alphaguess=LineSearches.InitialHagerZhang(),
        linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
res2 = Optim.optimize(f_composite, x0, LBFGS(;
        linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
# res3 = Optim.optimize(f_composite, x0, NewtonTrustRegion(), options)
