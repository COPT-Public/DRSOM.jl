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

include("../helper.jl")
include("../helper_plus.jl")
include("../helper_l.jl")
include("../helper_c.jl")
include("../helper_f.jl")
include("../helper_h.jl")
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
using .drsom_helper_l
using .drsom_helper_c
using .drsom_helper_f
using .hsodm_helper


using CUTEst
using NLPModels


# nlp = CUTEstModel("ARGLINA")
# nlp = CUTEstModel("BRYBND", "-param", "N=100")
# nlp = CUTEstModel("ARWHEAD", "-param", "N=500")
# nlp = CUTEstModel("CHAINWOO", "-param", "NS=49")
nlp = CUTEstModel("CHAINWOO", "-param", "NS=49")
name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
x0 = nlp.meta.x0
loss(x) = NLPModels.obj(nlp, x)
g(x) = NLPModels.grad(nlp, x)
H(x) = NLPModels.hess(nlp, x)


rh = hsodm_helper.run_drsomd(
    copy(x0), loss, g, H;
    maxiter=10000, tol=1e-8, freq=10,
    direction=:warm
)

rhc = hsodm_helper.run_drsomd(
    copy(x0), loss, g, H;
    maxiter=10000, tol=1e-7, freq=10,
    direction=:cold
)


method_objval_ragged = rstack([
        getresultfield.([rhc, rh], :kλ)...
    ]; fill=NaN
)
method_names = getname.([rhc, rh])


@printf("plotting results\n")

pgfplotsx()
title = L"\textrm{Warm-starting effect of eigenvector computation}"
fig = plot(
    1:(method_objval_ragged|>size|>first),
    method_objval_ragged,
    label=permutedims(method_names),
    xscale=:log10,
    yscale=:log10,
    xlabel="Iteration", ylabel=L"\textrm{Lanczos iterations to reach} \epsilon = 1e^{-8} ",
    title=title,
    size=(1280, 720),
    yticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2],
    xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
    dpi=1000,
)

savefig(fig, "/tmp/random-qp-$(n)-warmstart.pdf")

finalize(nlp)
