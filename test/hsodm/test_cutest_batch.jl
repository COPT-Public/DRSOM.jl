###############
# project=> RSOM
# created Date=> Tu Mar 2022
# author=> <<author>
# -----
# last Modified=> Mon Apr 18 2022
# modified By=> Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test HSODM Against Other methods
###############
include("./tools.jl")
include("./hsodm_paper_test.jl")

# include a small test to make sure everything works
@testset "TEST ALL DRSOM VARIANTS @ a CUTEst problem CHAINWOO" begin
    nlp = CUTEstModel("CHAINWOO", "-param", "NS=49")
    println(nlp.meta)
    name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
    x0 = nlp.meta.x0
    loss(x) = NLPModels.obj(nlp, x)
    g(x) = NLPModels.grad(nlp, x)
    H(x) = NLPModels.hess(nlp, x)

    # @testset "DRSOM" begin
    #     r = drsom_helper.run_drsomd(
    #         copy(x0), loss, g, H;
    #         maxiter=10000, tol=1e-6, freq=50
    #     )
    #     @test r.state.ϵ < 1e-4
    # end
    # @testset "DRSOM + Homogeneous Model" begin
    #     r = drsom_helper_plus.run_drsomd(
    #         copy(x0), loss, g, H;
    #         maxiter=10000, tol=1e-6, freq=50,
    #         direction=:homokrylov
    #     )
    #     @test r.state.ϵ < 1e-4
    # end
    @testset "HSODM" begin
        r = wrapper_hsodm(x0, loss, g, H)
        @test r.state.ϵ < 1e-4
    end
    finalize(nlp)
end


tables = []
header = ["name", "param", "n", "method", "k", "kf", "kg", "kh", "df", "fx", "t", "status", "update"]
fstamp = Dates.format(Dates.now(), dateformat"yyyy/mm/dd HH:MM:SS")
fstamppath = Dates.format(Dates.now(), dateformat"yyyymmddHHMM")
csvfile = open("cutest-$fstamppath.csv", "w")
write(csvfile, join(header, ","), "\n")
##########################################
# define your filters
##########################################
filter_cutest_problem(nlp) = true
# small test
# filter_cutest_problem(nlp) = (4 <= nlp.meta.nvar <= 200)
# large_test
# filter_cutest_problem(nlp) = (200 < nlp.meta.nvar <= 5000)

# filter_optimization_method(k) = k ∉ [:GD, :DRSOMHomo, :CG, :HSODM, :ARC]
# filter_optimization_method(k) = k == :CG
# filter_optimization_method(k) = k ∈ [:HSODM, :CG]
# filter_optimization_method(k) = k ∈ [:DRSOM, :CG]
# filter_optimization_method(k) = k == :DRSOM
# filter_optimization_method(k) = k ∈ [:DRSOM, :DRSOMHomo]
# filter_optimization_method(k) = k == :HSODM
filter_optimization_method(k) = k == :ARC
# filter_optimization_method(k) = k ∈ [:HSODM, :DRSOMHomo]
# filter_optimization_method(k) = k ∈ [:DRSOM, :HSODM, :DRSOMHomo]
# filter_optimization_method(k) = k ∈ [:HSODM, :DRSOMHomo, :LBFGS, :NewtonTR, :ARC]

# choose problem set
# PROBLEMS = UNC_PROBLEMS_221104
PROBLEMS = UNC_PROBLEMS_4to200
# PROBLEMS = UNC_PROBLEMS_200to5000

##########################################
# iteration
##########################################
# todo, add field kf, kg, kH, and inner iteration #
p = Progress(length(PROBLEMS); showspeed=true)
for (f, param_combination) in PROBLEMS
    for pc in param_combination
        try
            nlp = CUTEstModel(f, "-param", pc)
            name = "$(nlp.meta.name)-$(nlp.meta.nvar)"

            if !filter_cutest_problem(nlp)
                @warn("problem $name with $(nlp.meta.nvar) is not proper, skip")
                finalize(nlp)
                continue
            end
            @info("problem $name with $(nlp.meta.nvar) $name is good, continue")

            x0 = nlp.meta.x0
            loss(x) = NLPModels.obj(nlp, x)
            g(x) = NLPModels.grad(nlp, x)
            H(x) = NLPModels.hess(nlp, x)

            for (k, v) in OPTIMIZERS
                if !filter_optimization_method(k)
                    continue
                end
                @info("running $name $pc $k")
                line = []
                try
                    r = v(x0, loss, g, H)
                    line = [
                        nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        r.k, r.state.kf, r.state.kg, r.state.kH,
                        r.state.ϵ, r.state.fx, r.state.t, r.state.ϵ < 1e-5
                    ]

                catch e
                    line = [
                        nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        NaN, NaN, NaN, NaN,
                        NaN, NaN, NaN, false
                    ]

                    bt = catch_backtrace()
                    msg = sprint(showerror, e, bt)
                    println(msg)
                    @warn("instance $f opt $k failed")
                end
                # dump
                write(
                    csvfile,
                    join(line, ","),
                    ",",
                    fstamp,
                    "\n"
                )
                flush(csvfile)
            end
            for (k, v) in OPTIMIZERS_NLP
                if !filter_optimization_method(k)
                    continue
                end
                @info("running $name $pc $k")
                line = []
                try
                    r = v(nlp)
                    line = [
                        nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        r.k, r.state.kf, r.state.kg, r.state.kH,
                        r.state.ϵ, r.state.fx, r.state.t, r.state.ϵ < 1e-5
                    ]

                catch e
                    line = [
                        nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        NaN, NaN, NaN, NaN,
                        NaN, NaN, NaN, false
                    ]

                    bt = catch_backtrace()
                    msg = sprint(showerror, e, bt)
                    println(msg)
                    @warn("instance $f opt $k failed")
                end
                # dump
                write(
                    csvfile,
                    join(line, ","),
                    ",",
                    fstamp,
                    "\n"
                )
                flush(csvfile)
            end
            finalize(nlp)

        catch ef
            bt = catch_backtrace()
            msg = sprint(showerror, ef, bt)
            println(msg)
            @warn("instance $f loading failed")
            if isa(ef, InterruptException)
                @warn("user interrupted @ $f")
                exit(code=-1)
            end
        end
        ProgressMeter.next!(p)
        # @comment
        # only play with one that has proper size
        break
    end
end

close(csvfile)