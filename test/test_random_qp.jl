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

include("helper.jl")
include("lp.jl")

using ProximalOperators
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
using Optim
using .LP


params = LP.parse_commandline()
A, v, b = LP.create_random_lp(params)
x0 = zeros(params.m)

Q = A' * A
h = A' * b
# find Lipschitz constant
L, _ = LinearOperators.normest(Q, 1e-4)
# find convexity constant
# vals, vecs, info = KrylovKit.eigsolve(A'A, 1, :SR)
σ = 0.0
@printf("preprocessing finished\n")

########################################################
# direct evaluation
f_composite(x) = 1 / 2 * x' * Q * x - h' * x
g(x) = Q * x - h
H(x) = Q

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
name, state, k, traj = run_rsomd_traj(copy(x0), f_composite, g, H)