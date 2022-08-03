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
include("helper_plus.jl")

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
using MAT
using .LP
using .drsom_helper
using .drsom_helper_plus

data = matread("example/test.mat")
A = data["A"]
b = data["b"][:, 1]
f_composite(x) = 1 / 2 * x' * A * x - b' * x
g(x) = A * x - b
H(x) = A

x0 = zeros(100)
name, state, k, arr_obj = drsom_helper.run_drsomd(copy(x0), f_composite, g, H; maxiter=1000, tol=1e-6)
name, state, k, arr_obj = drsom_helper_plus.run_drsomd(copy(x0), f_composite, g, H; maxiter=1000, tol=1e-6)
