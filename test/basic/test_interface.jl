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
using Test
using .LP
using NLPModels
using CUTEst

params = LP.parse_commandline()
m = n = params.n
D = Normal(0.0, 1.0)
A = rand(D, (n, m)) .* rand(Bernoulli(0.85), (n, m))
v = rand(D, m) .* rand(Bernoulli(0.5), m)
b = A * v + rand(D, (n))
x0 = zeros(m)

Q = A' * A
h = Q' * b
L, _ = LinearOperators.normest(Q, 1e-4)
σ = 0.0
@printf("preprocessing finished\n")


f_composite(x) = 1 / 2 * x' * Q * x - h' * x
g(x) = Q * x - h
H(x) = Q


@testset "INTERFACE" begin
    @testset "DRSOM" begin
        alg = DRSOM2()
        @testset "direct hess" begin
            r = alg(x0=copy(x0), f=f_composite, g=g, H=H, fog=:direct, sog=:hess)
        end
        @testset "direct direct" begin
            r = alg(x0=copy(x0), f=f_composite, g=g, fog=:direct, sog=:direct)
        end

        @testset "forward direct" begin
            r = alg(x0=copy(x0), f=f_composite, fog=:forward, sog=:direct)
        end
        @testset "forward forward" begin
            r = alg(x0=copy(x0), f=f_composite, fog=:forward, sog=:forward)
        end
        @testset "backward direct" begin
            r = alg(x0=copy(x0), f=f_composite, fog=:backward, sog=:direct)
        end
        @testset "backward backward" begin
            r = alg(x0=copy(x0), f=f_composite, fog=:backward, sog=:backward)
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