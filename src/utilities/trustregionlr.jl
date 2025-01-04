
@doc raw"""
A simple procedure to solve TRS via a given regularizer 
!!! This is only used in DRSOM

We use this function (radius-free mode) `only` in DRSOM, 
    since the eigenvalues are easy to solve. 
"""
function SimpleTrustRegionSubproblem(
    Q,
    c::Vector{Float64},
    state::StateType;
    G=diagm(ones(2)),
    mode=:free,
    Δ::Float64=0.0,
    Δϵ::Float64=1e-4,
    Δl::Float64=1e2,
    λ::Float64=0.0
) where {StateType}

    # the dimension of Q and G should not be too big.
    _Q, _G = Symmetric(Q), Symmetric(G)
    _c = c

    if (mode == :free) && (length(_c) == 2)
        ######################################
        # the radius free mode
        ######################################
        # in this case, we know exactly the inverse
        # @special treatment for d = 2 (DRSOM 2D)
        a = _Q[1, 1]
        b = _Q[1, 2]
        d = _Q[2, 2]
        t = a + d
        s = a * d - b^2
        lmin = t / 2 - (t^2 / 4 - s)^0.5
        lmax = t / 2 + (t^2 / 4 - s)^0.5
        # eigvalues = eigvals(_Q, _G)
        # sort!(eigvalues)
        # lmin, lmax = eigvalues
        lb = max(1e-8, -lmin)
        ub = max(lb, lmax) + Δl
        state.λ = state.γ * lmax + max(1 - state.γ, 0) * lb
        _QG = _Q + state.λ .* _G
        a = _QG[1, 1]
        b = _QG[1, 2]
        d = _QG[2, 2]
        s = a * d - b^2
        K = [d/s -b/s; -b/s a/s]
        alpha = -K * _c
        return alpha
    end

    ######################################
    # compute eigenvalues for low-dimensional DRSOM
    ######################################
    eigvalues = begin
        try
            eigvals(_Q + 1e-10I, _G)
        catch e
            eigvals(_Q + 1e1I, _G)
        end
    end
    sort!(eigvalues)
    lmin, lmax = eigvalues[1], eigvalues[end]
    lb = max(1e-15, -lmin)
    ub = max(lb, lmax) + Δl
    if mode == :reg
        ######################################
        # the strict regularization mode
        ######################################
        # strictly solve a quadratic regularization problems
        #   given a regularizer :λ 
        ######################################
        state.λ = λ = state.γ * lmax + max(1 - state.γ, 0) * lb
        alpha = -(_Q + λ .* _G) \ _c
        return alpha
    end
    if mode == :trbisect
        ######################################
        # the strict radius mode
        ######################################
        # strictly solve TR given a radius :Δ 
        #   via bisection
        ######################################

        λ = lb
        try
            alpha = -(_Q + λ .* _G) \ _c
        catch e
            println(lb)
            printstyled(_Q)
            println(_Q + λ .* _G)
            throw(e)
        end

        s = sqrt(alpha' * _G * alpha) # size

        if s <= Δ
            # damped Newton step is OK
            state.λ = λ
            return alpha
        end
        # else we must hit the boundary
        while (ub - lb) > Δϵ
            λ = (lb + ub) / 2
            alpha = -(_Q + λ .* _G) \ _c
            s = sqrt(alpha' * _G * alpha) # size
            if s > Δ + Δϵ
                lb = λ
            elseif s < Δ - Δϵ
                ub = λ
            else
                # good enough
                break
            end
        end
        state.λ = λ
        return alpha
    end
    ex = ErrorException("Do not support the options $mode")
    throw(ex)
end



