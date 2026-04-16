using CUTEst
using Test
include("../tools.jl")
include("./problem_cols.jl")


log_freq = 200
precision = tol_grad = 1e-5
max_iter = 20000
max_time = 200.0
test_before_start = true


######################################################################
# include a small test to make sure everything works
# nlp = CUTEstModel("MSQRTALS", "-param", "P=7")
nlp = CUTEstModel("CHNROSNB", "-param", "N=25")
println(nlp.meta)
name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
x0 = nlp.meta.x0
loss(x) = NLPModels.obj(nlp, x)
g(x) = NLPModels.grad(nlp, x)
H(x) = NLPModels.hess(nlp, x)
hvp(x, v, Hv) = NLPModels.hprod!(nlp, x, v, Hv)


# @testset "DRSOM" begin
#     options_drsom = Dict(
#         :maxiter => max_iter,
#         :maxtime => max_time,
#         :tol => 1e-5,
#         :freq => log_freq
#     )
#     r = wrapper_drsom(x0, loss, g, H, options_drsom)
#     @test r.state.ϵ < 1e-4
# end
# @testset "DRSOM-d" begin
#     options_drsom = Dict(
#         :maxiter => max_iter,
#         :maxtime => max_time,
#         :tol => 1e-5,
#         :freq => log_freq
#     )
#     r = wrapper_drsomd(x0, loss, g, H, options_drsom; hvp=hvp)
#     @test r.state.ϵ < 1e-4
# end
@testset "HSODM" begin
    options = Dict(
        :maxiter => max_iter,
        :maxtime => max_time,
        :tol => 1e-5,
        :freq => log_freq
    )
    r = alg_hacubic_i(;
        x0=copy(x0), f=loss, g=g, H=H,
        A₀=1e-6,
        α=1.1,
        memory=10,
        memory_type=:ii,
        options...
    )
    @test r.state.ϵ < 1e-4
end
# @testset "UTR" begin
#     options_drsom = Dict(
#         :maxiter => max_iter,
#         :maxtime => max_time,
#         :tol => 1e-5,
#         :freq => log_freq
#     )
#     r = wrapper_utr(x0, loss, g, H, options_drsom)
#     @test r.state.ϵ < 1e-4
# end
# @testset "UTR" begin
#     options_drsom = Dict(
#         :maxiter => max_iter,
#         :maxtime => max_time,
#         :tol => 1e-5,
#         :freq => log_freq
#     )
#     r = wrapper_iutr_hvp(x0, loss, g, H, options_drsom; hvp=hvp)
#     @test r.state.ϵ < 1e-4
# end
finalize(nlp)