# support other packages
default_display(it, state::Any) =
    @printf("%5d | %.3e | %.3e | %.3e\n", it, state.f_x, state.gamma, norm(state.res, Inf))

default_display(it, obj::Real, step_size::Real, res::Real) =
    @printf("%5d | %.3e | %.3e | %.3e\n", it, obj, step_size, norm(res, Inf))

default_display(it, obj::Real, step_extra::Real, step_grad::Real, res::Real) =
    @printf("%5d | %.3e | %.3e | %.3e | %.3e\n", it, obj, step_extra, step_grad, norm(res, Inf))

default_stopping_criterion(tol, state::Any) = norm(state.res, Inf) <= tol

# formatter
const HEADER = [
    "The Dimension-Reduced Second-Order Method",
    "(c) Chuwen Zhang, Yinyu Ye, Cardinal Operations (2022)",
]

function format_header(log)
    loglength = log |> length
    sep = string(repeat("-", loglength))
    @printf("%s\n", sep)
    # print header lines in the middle
    for name in HEADER
        pref = loglength - (name |> length)
        prefs = string(repeat(" ", pref / 2 |> round |> Int))
        @printf("%s%s%s\n", prefs, name, prefs)
    end
    @printf("%s\n", sep)
end
