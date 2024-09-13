###############
# file: test_interface.jl
# author: Chuwen Zhang
# -----
# last Modified: Sat Dec 31 2022
# modified By: Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
###############


include("../lp.jl")

using ProximalOperators
using DRSOM
using ProximalAlgorithms
using Random

using Plots
using Printf
using LazyStack
using KrylovKit
using HTTP
using LaTeXStrings
using LinearAlgebra
using Statistics
using LinearOperators
using SparseArrays
using Optim
using Test
using .LP
using NLPModels
using CUTEst

params = LP.parse_commandline()
m = n = params.n
A = sprand(n, m, 0.85)
v = rand(m)
b = A * v + rand(n)
x0 = zeros(m)

Q = A' * A
h = Q' * b
L, _ = LinearOperators.normest(Q, 1e-4)
σ = 0.0
@printf("preprocessing finished\n")


f(x) = 1 / 2 * x' * Q * x - h' * x
g(x) = Q * x - h
hvp(x, v, buff) = copyto!(buff, Q * v)
H(x) = Q


@testset "INTERFACE" begin
    @testset "DRSOM" begin
        alg = DRSOM2()
        @testset "grad & hess" begin
            r = alg(x0=copy(x0), f=f, g=g, H=H, sog=:hess)
        end
        @testset "grad & direct" begin
            r = alg(x0=copy(x0), f=f, g=g, sog=:direct)
        end
        @testset "grad & hvp" begin
            r = alg(x0=copy(x0), f=f, g=g, hvp=hvp, sog=:hvp)
        end
        @testset "forward direct" begin
            r = alg(x0=copy(x0), f=f, fog=:forward, sog=:direct)
        end
        @testset "forward forward" begin
            r = alg(x0=copy(x0), f=f, fog=:forward, sog=:forward)
        end
        @testset "backward direct" begin
            r = alg(x0=copy(x0), f=f, fog=:backward, sog=:direct)
        end
        @testset "backward backward" begin
            r = alg(x0=copy(x0), f=f, fog=:backward, sog=:backward)
        end
    end

    @testset "HSODM" begin
        nlp = CUTEstModel("CHAINWOO", "-param", "NS=49")
        println(nlp.meta)
        name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
        x0 = nlp.meta.x0
        loss(x) = NLPModels.obj(nlp, x)
        g(x) = NLPModels.grad(nlp, x)
        H(x) = NLPModels.hess(nlp, x)
        alg = HSODM()
        for ls in (:trustregion, :hagerzhang)
            @testset "$ls" begin
                r = alg(x0=copy(x0), f=loss, g=g, H=H, linesearch=ls)
                @test r.state.ϵ < 1e-4
            end
        end
        finalize(nlp)
    end
end