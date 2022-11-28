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
# A script to test all DRSOM variants
###############

include("../helper.jl")
include("../helper_plus.jl")
include("../helper_l.jl")
include("../helper_c.jl")
include("../helper_f.jl")
include("../helper_h.jl")
include("../lp.jl")
using .LP
using .drsom_helper
using .drsom_helper_c
using .drsom_helper_f
using .drsom_helper_l
using .drsom_helper_plus
using .hsodm_helper
using ArgParse
using CUTEst
using DRSOM
using Distributions
using HTTP
using KrylovKit
using LaTeXStrings
using LazyStack
using LineSearches
using LinearAlgebra
using LinearOperators
using NLPModels
using Optim
using Plots
using Printf
using ProximalAlgorithms
using ProximalOperators
using Random
using Statistics
using Test



@testset "TEST ALL DRSOM VARIANTS @ a CUTEst problem CHAINWOO" begin
    nlp = CUTEstModel("CHAINWOO", "-param", "NS=49")
    println(nlp.meta)
    name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
    x0 = nlp.meta.x0
    loss(x) = NLPModels.obj(nlp, x)
    g(x) = NLPModels.grad(nlp, x)
    H(x) = NLPModels.hess(nlp, x)
    @testset "HSODM" begin
        r = hsodm_helper.run_drsomd(
            copy(x0), loss, g, H;
            maxiter=10000, tol=1e-8, freq=50,
            direction=:warm
        )
        @test r.state.ϵ < 1e-4
    end
    @testset "DRSOM" begin
        r = drsom_helper.run_drsomd(
            copy(x0), loss, g, H;
            maxiter=10000, tol=1e-6, freq=50
        )
        @test r.state.ϵ < 1e-4
    end
    @testset "DRSOM + Periodic Fix by Krylov" begin
        r = drsom_helper_f.run_drsomd(
            copy(x0), loss, g, H;
            maxiter=10000, tol=1e-6, freq=50,
            direction=:undef,
            direction_style=:truncate
        )
        @test r.state.ϵ < 1e-4
    end
    @testset "DRSOM + Curtis Style TR" begin
        r = drsom_helper_c.run_drsomd(
            copy(x0), loss, g, H;
            maxiter=10000, tol=1e-6, freq=50
        )
        @test r.state.ϵ < 1e-4
    end
    @testset "DRSOM + Homogeneous Model" begin
        r = drsom_helper_plus.run_drsomd(
            copy(x0), loss, g, H;
            maxiter=10000, tol=1e-6, freq=50,
            direction=:homokrylov
        )
        @test r.state.ϵ < 1e-4
    end
    finalize(nlp)
end