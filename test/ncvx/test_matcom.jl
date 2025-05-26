using DRSOM, DataFrames, CSV
using AdaptiveRegularization

include("./matcom.jl")
include("../tools.jl")

tables = []
for λ in [1e-2, 1e-3, 1e-4]
    for k = 1:1
        i = 1
        j = 1
        rows = 30
        columns = 48
        r = 9
        λ_1 = λ_2 = λ
        D, Ω = getData("test/instances", "Adamstown 132_11kV FY2021.csv", rows, columns, i, j)
        @time begin
            model = formulateMatrixCompletionProblem(D, Ω, r, λ_1, λ_2)
        end
        global nlp = MathOptNLPModel(model)


        x0 = nlp.meta.x0
        loss(x) = NLPModels.obj(nlp, x)
        g(x) = NLPModels.grad(nlp, x)
        H(x) = NLPModels.hess(nlp, x)
        hvp(x, v, Hv) = NLPModels.hprod!(nlp, x, v, Hv)

        ru = UTR(name=Symbol("Universal-TRS"))(;
            x0=copy(x0), f=loss, g=g, hvp=hvp,
            maxiter=300, tol=1e-5, freq=10,
            maxtime=1500,
            bool_trace=true,
            subpstrategy=:lanczos,
        )
        reset!(nlp)
        stats, _ = ARCqKOp(
            nlp,
            max_time=500.0,
            max_iter=500,
            max_eval=typemax(Int64),
            verbose=true
            # atol=atol,
            # rtol=rtol,
            # @note: how to set |g|?
        )
        rarc = arc_to_result(nlp, stats, "ARC")
        reset!(nlp)
        stats, _ = ST_TROp(
            nlp,
            max_time=500.0,
            max_iter=500,
            max_eval=typemax(Int64),
            verbose=true
            # atol=atol,
            # rtol=rtol,
            # @note: how to set |g|?
        )
        # AdaptiveRegularization.jl to my style of results
        rtrst = arc_to_result(nlp, stats, "TRST")

        finalize(nlp)
        push!(tables, [
            λ_1,
            "utr",
            ru.state.k,
            ru.state.kf,
            ru.state.kg,
        ])
        push!(tables, [
            λ_1,
            "arc",
            stats.iter,
            rarc.state.kf,
            rarc.state.kg,
        ])
        push!(tables, [
            λ_1,
            "trst",
            stats.iter,
            rtrst.state.kf,
            rtrst.state.kg,
        ])
    end
end
df = DataFrame(hcat(tables...)', [:λ, :name, :k, :kf, :kg])

CSV.write("1.csv", df)
