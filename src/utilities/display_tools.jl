default_display(it, state::Any) =
    @printf("%5d | %.3e | %.3e | %.3e\n", it, state.f_x, state.gamma, norm(state.res, Inf))

default_display(it, obj::Real, step_size::Real, res::Real) =
    @printf("%5d | %.3e | %.3e | %.3e\n", it, obj, step_size, norm(res, Inf))

default_display(it, obj::Real, step_extra::Real, step_grad::Real, res::Real) =
    @printf("%5d | %.3e | %.3e | %.3e | %.3e\n", it, obj, step_extra, step_grad, norm(res, Inf))

default_stopping_criterion(tol, state::Any) = norm(state.res, Inf) <= tol
