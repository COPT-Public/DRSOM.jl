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

Random.seed!(2)

for n in [100, 200, 1000]
    for m in [10, 20, 100]
        p = 0.5
        D = Uniform(0.0, 1.0)
        A = rand(D, (n, m)) .* rand(Bernoulli(0.15), (n, m))
        v = rand(D, m) .* rand(Bernoulli(0.5), m)
        b = A * v + rand(D, (n))
        x0 = zeros(m)

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
        f_composite(x) = 1 / 2 * (A * x - b)' * (A * x - b) + LP.huberlike(λ, 0.1, p, x)
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

        # rsom
        name, state, k, arr_obj = run_rsomb(
            copy(x0), f_composite;
            tol=1e-7,
            maxiter=1000,
            freq=10000000
        )
        method_objval[name] = copy(arr_obj)
        iter_scale = max(iter_scale, k) # compute max plot scale

        # compare with GD and LBFGS, Trust region newton,
        options = Optim.Options(
            g_tol=1e-5,
            iterations=1000,
            store_trace=true,
            show_trace=true
        )
        res1 = Optim.optimize(f_composite, x0, GradientDescent(;
                alphaguess=LineSearches.InitialHagerZhang(),
                linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
        res2 = Optim.optimize(f_composite, x0, LBFGS(;
                linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
        res3 = Optim.optimize(f_composite, x0, NewtonTrustRegion(), options)

        @printf("tabular start\n")
        @printf("%d,%d,%d,%.4e,%.4e,%.3f\n", n, m, k, state.fx, norm(state.∇f), state.t)
        @printf("%d,%d,%d,%.4e,%.4e,%.3f\n", n, m, res1.trace[end].iteration, res1.trace[end].value, res1.trace[end].g_norm, res1.time_run)
        @printf("%d,%d,%d,%.4e,%.4e,%.3f\n", n, m, res2.trace[end].iteration, res2.trace[end].value, res2.trace[end].g_norm, res2.time_run)
        @printf("%d,%d,%d,%.4e,%.4e,%.3f\n", n, m, res3.trace[end].iteration, res3.trace[end].value, res3.trace[end].g_norm, res3.time_run)
        @printf("tabular end\n")
        @printf("summary start\n")
        @printf("==%d,%d,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f\n", n, m,
            k, state.fx, norm(state.∇f), state.t,
            res1.trace[end].iteration, res1.trace[end].value, res1.trace[end].g_norm, res1.time_run,
            res2.trace[end].iteration, res2.trace[end].value, res2.trace[end].g_norm, res2.time_run,
            res3.trace[end].iteration, res3.trace[end].value, res3.trace[end].g_norm, res3.time_run
        )
        @printf("summary end")

    end
end