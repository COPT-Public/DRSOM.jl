
################################################################################
# NEWTON STEPS
################################################################################
function NewtonStep(
    H::SparseMatrixCSC{R,T}, μ, g, state; verbose::Bool=false
) where {R<:Real,T<:Int}
    # d, __unused_info = KrylovKit.linsolve(
    #     H + μ * SparseArrays.I, -Vector(g);
    #     isposdef=true, issymmetric=true
    # )
    # d = -((H + μ * I) \ g)
    cc = ldl(H + μ * I)
    d = -cc \ g
    return 1, 1, d, norm(d), d' * state.∇f, d' * H * d
end

function NewtonStep(H::Matrix{R}, μ, g, state; verbose::Bool=false
) where {R<:Real}
    d = -((H + μ * I) \ g)
    return 1, 1, d, norm(d), d' * state.∇f, d' * H * d
end

@doc """
@note, one can use cg from `KrylovKit.jl`, to me it seems to be the same
`Krylov.jl` seems to be a little faster
```julia
d, __unused_info = KrylovKit.linsolve(
    f, -Vector(g);
    isposdef=true, issymmetric=true
)
```
"""
function NewtonStep(iter::I, μ, g, state; verbose::Bool=false
) where {I}
    @debug "started Newton-step @" Dates.now()

    n = g |> length
    gn = (g |> norm)
    # opH = LinearOperator(Float64, n, n, true, true, (y, v) -> iter.ff(y, v))
    # d, _info = cg(
    #     opH, -Vector(g);
    #     # rtol=state.ϵ > 1e-4 ? 1e-7 : iter.eigtol, 
    #     rtol=state.ϵ > 1e-4 ? gn * 1e-4 : gn * 1e-6,
    #     itmax=200,
    #     verbose=verbose ? 3 : 0
    # )
    # return _info.niter, 1, d, norm(d), d' * state.∇f, d' * state.∇fb
    f(v) = iter.ff(state.∇fb, v)
    d, _info = KrylovKit.linsolve(
        f, -Vector(g), -Vector(g),
        CG(;
            tol=min(gn, 1e-4),
            maxiter=n * 2,
            verbosity=verbose ? 3 : 0
        ),
    )
    return _info.numops, 1, d, norm(d), d' * state.∇f, d' * state.∇fb
end

function NewtonStep(opH::LinearOperator, g, state; verbose::Bool=false)
    # @debug "started Newton-step @" Dates.now()
    n = g |> length
    gn = (g |> norm)
    # d, _info = cg(
    #     opH, -Vector(g);
    #     rtol=state.ϵ > 1e-4 ? gn * 1e-4 : gn * 1e-6,
    #     itmax=200,
    #     verbose=verbose ? 3 : 0
    # )
    # return _info.niter, 1, d, norm(d), d' * state.∇f, d' * state.∇fb
    f(v) = iter.ff(state.∇fb, v)
    d, _info = KrylovKit.linsolve(
        f, -Vector(g), -Vector(g),
        CG(;
            tol=min(gn, 1e-4),
            maxiter=n * 2,
            verbosity=verbose ? 3 : 0
        ),
    )
end