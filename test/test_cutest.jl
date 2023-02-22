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

include("tools.jl")

using ProximalOperators
using LineSearches
using Optim
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
using CUTEst
using NLPModels

using DRSOM

#######################################################
# examples
# nlp = CUTEstModel("BROYDN7D", "-param", "N/2=2500")
# nlp = CUTEstModel("SSBRYBND", "-param", "N=50")
# nlp = CUTEstModel("SCURLY10", "-param", "N=10")
# nlp = CUTEstModel("ARGLINA", "-param", "M=200,N=200")
# nlp = CUTEstModel("BRYBND", "-param", "N=100")
# nlp = CUTEstModel("BRYBND", "-param", "N=100")
# nlp = CUTEstModel("EXTROSNB", "-param", "N=100")
# nlp = CUTEstModel("CURLY10", "-param", "N=100")
# nlp = CUTEstModel("CRAGGLVY", "-param", "M=24")
# nlp = CUTEstModel("ARWHEAD", "-param", "N=500")
# nlp = CUTEstModel("COSINE", "-param", "N=100")
# nlp = CUTEstModel("CHAINWOO", "-param", "NS=49")
# nlp = CUTEstModel("BIGGS6", "-param", "NS=49")
# nlp = CUTEstModel("FMINSRF2", "-param", "NS=49")
#######################################################

bool_plotting = true
# name, param = ARGS[1:2]
# name, param = ["CHAINWOO" "NS=49"]
# nlp = CUTEstModel(name, "-param", param)

function get_needed_entry(r)
    return @sprintf("%d,%.1e,%.3f", r.k, r.state.ϵ, r.state.t)
end

function get_needed_entry_optim(r)
    return @sprintf("%d,%.1e,%.3f", r.k, r.state.ϵ, r.state.t)
end



name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
x0 = nlp.meta.x0
loss(x) = NLPModels.obj(nlp, x)
g(x) = NLPModels.grad(nlp, x)
H(x) = NLPModels.hess(nlp, x)


# compare with GD and LBFGS, Trust region newton,
options = Optim.Options(
    g_tol=1e-6,
    iterations=1000,
    store_trace=true,
    show_trace=true,
    show_every=1,
    time_limit=500
)

# res1 = Optim.optimize(loss, g, x0,
#     GradientDescent(;
#         alphaguess=LineSearches.InitialHagerZhang(),
#         linesearch=LineSearches.StrongWolfe()
#     ), options; inplace=false)
# res2 = Optim.optimize(loss, g, H, x0,
#     LBFGS(;
#         linesearch=LineSearches.StrongWolfe()
#     ), options; inplace=false)
# res3 = Optim.optimize(loss, g, H, x0,
#     NewtonTrustRegion(
#     ), options; inplace=false)

# res4 = Optim.optimize(loss, g, H, x0,
#     ConjugateGradient(;
#         alphaguess=LineSearches.InitialStatic(),
#         linesearch=LineSearches.HagerZhang()
#     ), options; inplace=false)

# # arc = wrapper_arc(nlp)

# r = DRSOM2()(;
#     x0=copy(x0), f=loss, g=g,
#     maxiter=10000, tol=1e-6, freq=1
# )

# rh = HSODM(; name=:HSODMLS)(;
#     x0=copy(x0), f=loss, g=g, H=H,
#     maxiter=10000, tol=1e-6, freq=1,
#     direction=:warm, linesearch=:hagerzhang
# )
rha = HSODM(; name=:HSODMArC)(;
    x0=copy(x0), f=loss, g=g, H=H,
    maxiter=10000, tol=1e-6, freq=1,
    direction=:warm, adaptive=:arc,
    maxtime=10000
)

finalize(nlp)

# rarc = wrapper_arc(nlp)
# results = [
#     # optim_to_result(res1, "GD+Wolfe"),
#     optim_to_result(res2, "LBFGS+Wolfe"),
#     optim_to_result(res3, "Newton-TR"),
#     # optim_to_result(res4, "CG"),
#     # arc,
#     # r,
#     rh,
#     rha,
# ]

# if bool_plotting

#     for metric in (:fx, :ϵ)
#         method_objval_ragged = rstack([
#                 getresultfield.(results, metric)...
#             ]; fill=NaN
#         )
#         method_names = getname.(results)


#         @printf("plotting results\n")

#         pgfplotsx()
#         title = "CUTEst model name := $(name)"
#         fig = plot(
#             1:(method_objval_ragged|>size|>first),
#             method_objval_ragged,
#             label=permutedims(method_names),
#             xscale=:log2,
#             yscale=:log10,
#             xlabel="Iteration",
#             ylabel=metric == :ϵ ? L"\|\nabla f\| = \epsilon" : L"f(x)",
#             title=title,
#             size=(1280, 720),
#             yticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2],
#             xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
#             dpi=1000,
#         )

#         savefig(fig, @sprintf("/tmp/CUTEst-%s-%s.pdf", name, metric))
#     end

# end

