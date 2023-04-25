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

using CUTEst
using Test
include("./test_setup.jl")


tables = []
header = [
    "precision", "name", "param", "n", "method",
    "k", "kf", "kg", "kh", "df", "fx", "t", "status", "update"
]
fstamp = Dates.format(Dates.now(), dateformat"yyyy/mm/dd HH:MM:SS")
fstamppath = Dates.format(Dates.now(), dateformat"yyyymmddHHMM")
csvfile = open("cutest-$fstamppath.csv", "w")
write(csvfile, join(header, ","), "\n")


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
            hvp(x, v, Hv) = NLPModels.hprod!(nlp, x, v, Hv)
            # compute g(x₀)
            g₀ = g(x0) |> norm
            # if too large then stop at a relative measure
            this_tol = g₀ > 1e10 ? tol_grad * g₀ : tol_grad
            for (k, v) in MY_OPTIMIZERS

                # general options for my optimizers,
                options_drsom = Dict(
                    :maxiter => max_iter,
                    :maxtime => max_time,
                    :tol => this_tol,
                    :freq => log_freq
                )
                if !filter_optimization_method(k)
                    continue
                end
                @info("running $name $pc $k")
                line = []
                try
                    r = v(x0, loss, g, H, options_drsom; hvp=hvp)
                    line = [
                        precision, nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        r.k, r.state.kf, r.state.kg + r.state.kh, r.state.kH,
                        r.state.ϵ, r.state.fx, r.state.t,
                        min(r.state.ϵ, r.state.ϵ / g₀) < precision
                    ]

                catch e
                    line = [
                        precision, nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
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
            for (k, v) in OPTIMIZERS_OPTIM
                # general options for Optim
                # GD and LBFGS, Trust Region Newton,
                options = Optim.Options(
                    g_tol=this_tol,
                    iterations=max_iter,
                    store_trace=true,
                    show_trace=true,
                    show_every=log_freq,
                    time_limit=max_time
                )
                if !filter_optimization_method(k)
                    continue
                end
                @info("running $name $pc $k")
                line = []
                try
                    r = v(x0, loss, g, H, options; hvp=hvp)
                    line = [
                        precision, nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        r.k, r.state.kf, r.state.kg + r.state.kh, r.state.kH,
                        r.state.ϵ, r.state.fx, r.state.t,
                        min(r.state.ϵ, r.state.ϵ / g₀) < precision
                    ]

                catch e
                    line = [
                        precision, nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
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
                        precision, nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
                        r.k, r.state.kf, r.state.kg + r.state.kh, r.state.kH,
                        r.state.ϵ, r.state.fx, r.state.t,
                        min(r.state.ϵ, r.state.ϵ / g₀) < precision
                    ]

                catch e
                    line = [
                        precision, nlp.meta.name, "\"$pc\"", nlp.meta.nvar, k,
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
        flush(stdout)
        # @comment
        # only play with one that has proper size
        break
    end
end

close(csvfile)