include("../tools.jl")
include("./hsodm_paper_test.jl")


log_freq = 1
precision = tol_grad = 1e-5
max_iter = 20000
max_time = 200.0
test_before_start = false


##########################################
# define at run time
##########################################
# test set
filter_cutest_problem(nlp) = true
# small test
# filter_cutest_problem(nlp) = (4 <= nlp.meta.nvar <= 200)
# large_test
# filter_cutest_problem(nlp) = (200 < nlp.meta.nvar <= 5000)


# filter_optimization_method(k) = true
# filter_optimization_method(k) = k == :CG
# filter_optimization_method(k) = k ∉ [:GD, :DRSOMHomo, :CG, :HSODM, :ARC]
# filter_optimization_method(k) = k == :CG
# filter_optimization_method(k) = k ∈ [:HSODM, :DRSOM, :CG]
# filter_optimization_method(k) = k ∈ [:DRSOM, :CG]
# filter_optimization_method(k) = k ∈ [:LBFGS, :NewtonTR]
# filter_optimization_method(k) = k ∈ [:LBFGS]
# filter_optimization_method(k) = k == :DRSOM
# filter_optimization_method(k) = k == :LBFGS
# filter_optimization_method(k) = k ∈ [:LBFGS, :HSODM]
# filter_optimization_method(k) = k ∈ [:DRSOM, :DRSOMHomo]
# filter_optimization_method(k) = k == :HSODM
filter_optimization_method(k) = k ∈ [:iUTR]
# filter_optimization_method(k) = k == :ARC
# filter_optimization_method(k) = k == :HSODMArC
# filter_optimization_method(k) = k ∈ [:HSODM, :DRSOMHomo]
# filter_optimization_method(k) = k ∈ [:DRSOM, :HSODM, :DRSOMHomo]
# filter_optimization_method(k) = k ∈ [:HSODM, :DRSOMHomo, :LBFGS, :NewtonTR, :ARC]
# filter_optimization_method(k) = k ∈ [:NewtonTR, :ARC]

# choose problem set
# PROBLEMS = UNC_PROBLEMS_221104
# PROBLEMS = TEST
# PROBLEMS = UNC_PROBLEMS_4to200
PROBLEMS = UNC_PROBLEMS_200to5000

if test_before_start
    ######################################################################
    # include a small test to make sure everything works
    @testset "TEST ALL DRSOM VARIANTS @ a CUTEst problem CHAINWOO" begin
        nlp = CUTEstModel("MSQRTALS", "-param", "P=7")
        println(nlp.meta)
        name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
        x0 = nlp.meta.x0
        loss(x) = NLPModels.obj(nlp, x)
        g(x) = NLPModels.grad(nlp, x)
        H(x) = NLPModels.hess(nlp, x)
        hvp(x, v, Hv) = NLPModels.hprod!(nlp, x, v, Hv)


        @testset "DRSOM" begin
            options_drsom = Dict(
                :maxiter => max_iter,
                :maxtime => max_time,
                :tol => 1e-5,
                :freq => log_freq
            )
            r = wrapper_drsom(x0, loss, g, H, options_drsom)
            @test r.state.ϵ < 1e-4
        end
        @testset "DRSOM-d" begin
            options_drsom = Dict(
                :maxiter => max_iter,
                :maxtime => max_time,
                :tol => 1e-5,
                :freq => log_freq
            )
            r = wrapper_drsomd(x0, loss, g, H, options_drsom; hvp=hvp)
            @test r.state.ϵ < 1e-4
        end
        @testset "HSODM" begin
            options_drsom = Dict(
                :maxiter => max_iter,
                :maxtime => max_time,
                :tol => 1e-5,
                :freq => log_freq
            )
            r = wrapper_hsodm(x0, loss, g, H, options_drsom)
            @test r.state.ϵ < 1e-4
        end
        finalize(nlp)
    end
end