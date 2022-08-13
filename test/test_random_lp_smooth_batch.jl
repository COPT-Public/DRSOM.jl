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
include("helper_plus.jl")
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
using .drsom_helper
using .drsom_helper_plus

Random.seed!(2)
logger = []
tables = []
# for n in [100, 200, 1000, 10000]
#     for m in [50, 100, 1000, 5000]
for n in [50, 100, 1000]
    for m in [50, 100, 200]
        for nnz in [0.2, 0.6, 0.8]
            for idx in 1:2
                p = 0.5
                D = Uniform(0.0, 1.0)
                A = rand(D, (n, m)) .* rand(Bernoulli(nnz), (n, m))
                v = rand(D, m) .* rand(Bernoulli(nnz), m)
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
                f = ProximalOperators.LeastSquares(A, b)
                # direct evaluation
                # smooth 1
                # f_composite(x) = 1 / 2 * (A * x - b)' * (A * x - b) + LP.smoothlp(λ, 0.1, params.p, x)
                # g(x) = A' * (A * x - b) + LP.smoothlpg(λ, 0.1, params.p, x)
                # H(x) = A' * A + LP.smoothlph(λ, 0.1, params.p, x)

                # huberlike
                f_composite(x) = 1 / 2 * (A * x - b)' * (A * x - b) + LP.huberlike(λ, 0.1, p, x)
                g(x) = A' * (A * x - b) + LP.huberlikeg(λ, 0.1, p, x)
                H(x) = A' * A + LP.huberlikeh(λ, 0.1, p, x)

                lowtol = 1e-6
                tol = 1e-6
                maxtime = 300
                maxiter = 2000
                verbose = true
                freq = 40

                #######
                iter_scale = 0
                #######

                method_objval = Dict{String,AbstractArray{Float64}}()
                method_state = Dict{String,Any}()

                # drsom
                name, state, k, arr_obj = drsom_helper.run_drsomd(
                    copy(x0), f_composite, g, H;
                    tol=1e-7,
                    maxiter=maxiter,
                    freq=10
                )
                # drsom krylov
                namek, statek, kk, arr_objk = drsom_helper_plus.run_drsomd(
                    copy(x0), f_composite, g, H;
                    tol=1e-7,
                    maxiter=maxiter,
                    freq=10,
                    direction=:krylov
                )
                # drsom gaussian no.1
                nameg1, stateg1, kg1, arr_objg1 = drsom_helper_plus.run_drsomd(
                    copy(x0), f_composite, g, H;
                    tol=1e-7,
                    maxiter=maxiter,
                    freq=10,
                    direction=:gaussian,
                    direction_num=1
                )
                # drsom gaussian no.3
                nameg3, stateg3, kg3, arr_objg3 = drsom_helper_plus.run_drsomd(
                    copy(x0), f_composite, g, H;
                    tol=1e-7,
                    maxiter=maxiter,
                    freq=10,
                    direction=:gaussian,
                    direction_num=3
                )

                # drsom gaussian no.5
                nameg5, stateg5, kg5, arr_objg5 = drsom_helper_plus.run_drsomd(
                    copy(x0), f_composite, g, H;
                    tol=1e-7,
                    maxiter=maxiter,
                    freq=10,
                    direction=:gaussian,
                    direction_num=10
                )



                method_objval[name] = copy(arr_obj)
                iter_scale = max(iter_scale, k) # compute max plot scale

                # compare with GD and LBFGS, Trust region newton,
                options = Optim.Options(
                    g_tol=1e-5,
                    iterations=maxiter,
                    store_trace=true,
                    show_trace=true,
                    show_every=20
                )
                # res1 = Optim.optimize(f_composite, x0, GradientDescent(;
                #         alphaguess=LineSearches.InitialHagerZhang(),
                #         linesearch=LineSearches.StrongWolfe()), options; autodiff=:forward)
                # res3 = Optim.optimize(f_composite, x0, NewtonTrustRegion(), options)
                res2 = Optim.optimize(
                    f_composite,
                    g,
                    H,
                    x0,
                    LBFGS(; linesearch=LineSearches.StrongWolfe()),
                    options;
                    inplace=false
                )
                size = @sprintf("%d,%d,%.2f,%d", n, m, nnz, idx)

                push!(tables, string(size, ",DRSOM,", @sprintf("%d,%.4e,%.4e,%.3f\n", k, state.fx, norm(state.∇f), state.t)))
                push!(tables, string(size, ",DRSOM-Krylov,", @sprintf("%d,%.4e,%.4e,%.3f\n", kk, statek.fx, norm(statek.∇f), statek.t)))
                push!(tables, string(size, ",DRSOM-Gaussian-1d,", @sprintf("%d,%.4e,%.4e,%.3f\n", kg1, stateg1.fx, norm(stateg1.∇f), stateg1.t)))
                push!(tables, string(size, ",DRSOM-Gaussian-3d,", @sprintf("%d,%.4e,%.4e,%.3f\n", kg3, stateg3.fx, norm(stateg3.∇f), stateg3.t)))
                push!(tables, string(size, ",DRSOM-Gaussian-10d,", @sprintf("%d,%.4e,%.4e,%.3f\n", kg5, stateg5.fx, norm(stateg5.∇f), stateg5.t)))
                # @printf("%d,%d,%d,%.4e,%.4e,%.3f\n", n, m, res1.trace[end].iteration, res1.trace[end].value, res1.trace[end].g_norm, res1.time_run)
                # @printf("%d,%d,%d,%.4e,%.4e,%.3f\n", n, m, res3.trace[end].iteration, res3.trace[end].value, res3.trace[end].g_norm, res3.time_run)
                push!(tables, string(size, ",LBFGS,", @sprintf("%d,%.4e,%.4e,%.3f\n", res2.trace[end].iteration, res2.trace[end].value, res2.trace[end].g_norm, res2.time_run)))

                # @printf("==%d,%d,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f,%d,%.4e,%.4e,%.3f\n",

                data = [
                    size,
                    # DRSOM-original
                    @sprintf("%d,%.4e,%.4e,%.3f", k, state.fx, norm(state.∇f), state.t),
                    # DRSOM-krylov
                    @sprintf("%d,%.4e,%.4e,%.3f", kk, statek.fx, norm(statek.∇f), statek.t),
                    # # DRSOM-Gaussian
                    @sprintf("%d,%.4e,%.4e,%.3f", kg1, stateg1.fx, norm(stateg1.∇f), stateg1.t),
                    @sprintf("%d,%.4e,%.4e,%.3f", kg3, stateg3.fx, norm(stateg3.∇f), stateg3.t),
                    @sprintf("%d,%.4e,%.4e,%.3f", kg5, stateg5.fx, norm(stateg5.∇f), stateg5.t),
                    # # LBFGS
                    @sprintf("%d,%.4e,%.4e,%.3f", res2.trace[end].iteration, res2.trace[end].value, res2.trace[end].g_norm, res2.time_run),
                    # AGD
                    # @sprintf("%d,%.4e,%.4e,%.3f" res1.trace[end].iteration res1.trace[end].value res1.trace[end].g_norm res1.time_run,
                    # Newton TR
                    # @sprintf("%d,%.4e,%.4e,%.3f" res3.trace[end].iteration res3.trace[end].value res3.trace[end].g_norm res3.time_run,
                    # tail
                ]
                line = join(data, ",")
                display(line)
                @printf("summary end\n")
                push!(logger, line)
            end
        end
    end
end

for l in logger
    @printf "%s\n" l
end


@printf "%s\n" repeat("+", 100)
for l in tables
    @printf "%s" l
end