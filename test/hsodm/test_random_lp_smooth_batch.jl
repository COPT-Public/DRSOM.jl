###############
# project: DRSOM
# created Date: Tu Mar 2022
# author: <<author>
# -----
# last Modified: Mon Apr 18 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test on smoothed L2-Lp minimization problems,
# Comparison of DRSOM and A "real" second-order mothod (Newton-trust-region)
# For L2-Lp minimization, see the paper by X. Chen
# 1. Chen, X.: Smoothing methods for nonsmooth, nonconvex minimization. Math. Program. 134, 71–99 (2012). https://doi.org/10.1007/s10107-012-0569-0
# 2. Chen, X., Ge, D., Wang, Z., Ye, Y.: Complexity of unconstrained $$L_2-L_p$$ minimization. Math. Program. 143, 371–383 (2014). https://doi.org/10.1007/s10107-012-0613-0
# 3. Ge, D., Jiang, X., Ye, Y.: A note on the complexity of Lp minimization. Mathematical Programming. 129, 285–299 (2011). https://doi.org/10.1007/s10107-011-0470-2
###############

include("../helper.jl")
include("../helper_h.jl")
include("../helper_plus.jl")
include("../lp.jl")

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
using .drsom_helper
using .drsom_helper_plus
using .hsodm_helper

function get_needed_entry(r)
    return @sprintf("%d,%.1e,%.3f", r.k, r.state.ϵ, r.state.t)
end
function get_needed_entry_optim(r)
    return @sprintf("%d,%.1e,%.3f", r.trace[end].iteration, r.trace[end].g_norm, r.time_run)
end

Random.seed!(2)
logger = []
tables = []
# for n in [100, 200, 1000, 10000]
#     for m in [50, 100, 1000, 5000]
# for n in [50]
#     for m in [50]
#         for nnz in [0.2]
#             for idx in [1]
for nnz in [0.15, 0.25]
    for n in [300, 500, 1000]
        for m in [100, 200, 500]
            for idx in 1:1
                p = 0.5
                D = Uniform(0.0, 1.0)
                A = rand(D, (n, m)) .* rand(Bernoulli(nnz), (n, m))
                v = rand(D, m) .* rand(Bernoulli(0.5), m)
                b = A * v + rand(D, (n))
                x0 = zeros(m)

                gram = A' * A

                # find Lipschitz constant
                L, _ = LinearOperators.normest(gram, 1e-4)
                λ = LinearAlgebra.normInf(A'b) / 3
                # find convexity constant
                # vals, vecs, info = KrylovKit.eigsolve(A'A, 1, :SR)
                σ = 0.0
                @printf("preprocessing finished\n")

                ########################################################
                # direct evaluation
                # smooth 1
                # f_composite(x) = 1 / 2 * (A * x - b)' * (A * x - b) + LP.smoothlp(λ, 0.1, params.p, x)
                # g(x) = A' * (A * x - b) + LP.smoothlpg(λ, 0.1, params.p, x)
                # H(x) = A' * A + LP.smoothlph(λ, 0.1, params.p, x)

                # huberlike
                f_composite(x) = 1 / 2 * (A * x - b)' * (A * x - b) + LP.huberlike(λ, 0.1, p, x)
                g(x) = A' * (A * x - b) + LP.huberlikeg(λ, 0.1, p, x)
                H(x) = A' * A + LP.huberlikeh(λ, 0.1, p, x)

                kwds = Dict(
                    :tol => 1e-5,
                    :maxiter => 10000,
                    :freq => 40
                )
                options = Optim.Options(
                    g_tol=1e-5,
                    iterations=10000,
                    store_trace=true,
                    show_trace=true,
                    show_every=40
                )
                # drsom
                r = drsom_helper.run_drsomd(
                    copy(x0), f_composite, g, H;
                    kwds...
                )

                # drsom-h
                rdh = drsom_helper_plus.run_drsomd(
                    copy(x0), f_composite, g, H;
                    direction=:homokrylov,
                    kwds...
                )

                # hsodm
                rh = hsodm_helper.run_drsomd(
                    copy(x0), f_composite, g, H;
                    direction=:warm,
                    kwds...
                )

                # compare with GD and LBFGS, Trust region newton,
                res1 = Optim.optimize(f_composite,
                    g,
                    x0,
                    ConjugateGradient(),
                    options;
                    inplace=false
                )
                res2 = Optim.optimize(
                    f_composite,
                    g,
                    H,
                    x0,
                    LBFGS(; linesearch=LineSearches.StrongWolfe()),
                    options;
                    inplace=false
                )
                res3 = Optim.optimize(
                    f_composite,
                    g,
                    H,
                    x0,
                    NewtonTrustRegion(),
                    options;
                    inplace=false
                )
                size = @sprintf("%d,%d,%.2f,%d", n, m, nnz, idx)

                # SQL like structure
                # push!(tables, push_state_to_string(size, r.state, "DRSOM"))
                # push!(tables, push_state_to_string(size, rh.state, "HSODM"))
                # push!(tables, push_optim_result_to_string(size, res1, "CG"))
                # push!(tables, push_optim_result_to_string(size, res2, "BFGS"))
                # push!(tables, push_optim_result_to_string(size, res2, "Newton-TR"))

                # LaTeX table like structure

                data = [
                    size,
                    get_needed_entry(r),
                    get_needed_entry(rh),
                    get_needed_entry(rdh),
                    get_needed_entry_optim(res1),
                    get_needed_entry_optim(res2),
                    get_needed_entry_optim(res3)
                ]
                line = join(data, ",")
                display(line)
                @printf("summary end\n")
                push!(logger, line)
            end
        end
    end
end
@printf "%s\n" repeat("+", 100)
for l in tables
    @printf "%s" l
end
@printf "%s\n" repeat("+", 100)
for l in logger
    @printf "%s\n" l
end

