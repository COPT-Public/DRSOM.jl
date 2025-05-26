include("./hagerzhang.jl")
include("./backtracking.jl")

const lsa::HagerZhangEx = HagerZhangEx(
# linesearchmax=10
)
const lsb::BackTrackingEx = BackTrackingEx(
    ρ_hi=0.8,
    ρ_lo=0.1,
    order=3
)

function TRStyleLineSearch(
    iter::IterationType,
    z::Tx,
    s::Tg,
    vHv::Real,
    vg::Real,
    f₀::Real,
    γ₀::Real=1.0;
    ρ::Real=0.7,
    ψ::Real=0.8
) where {IterationType,Tx,Tg}
    it = 1
    γ = γ₀
    fx = f₀
    while true
        # summarize
        x = z + s * γ
        fx = iter.f(x)
        dq = -γ^2 * vHv / 2 - γ * vg
        df = f₀ - fx
        ro = df / dq
        acc = (ro > ρ && df > 0) || it <= iter.itermax
        if !acc
            γ *= ψ
        else
            break
        end
        it += 1
    end
    return γ, fx, it
end


function BacktrackLineSearch(
    iter::IterationType,
    gx, fx,
    x::Tx,
    s::Tg
) where {IterationType,Tx,Tg}

    ϕ(α) = iter.f(x .+ α .* s)
    function dϕ(α)
        if iter.g !== nothing
            gv = iter.g(x + α .* s)
        else
            gv = similar(s)
            iter.ga(gv, x + α .* s)
        end
        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = iter.f(x .+ α .* s)
        if iter.g !== nothing
            gv = iter.g(x + α .* s)
        else
            gv = similar(s)
            iter.ga(gv, x + α .* s)
        end
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)

    try
        α, fx, it = lsb(ϕ, dϕ, ϕdϕ, 4.0, fx, dϕ_0)
        return α, fx, it
    catch y
        throw(y)
        isa(y, LineSearchException) # && println() # todo

        return 0.1, fx, 1
    end
end

function BacktrackLineSearch(
    f, g,
    gx, fx,
    x::Tx,
    s::Tg
) where {Tx,Tg}

    ϕ(α) = f(x .+ α .* s)
    function dϕ(α)

        gv = g(x + α .* s)

        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = f(x .+ α .* s)
        gv = g(x + α .* s)
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)
    try
        α, fx = lsb(ϕ, dϕ, ϕdϕ, 4.0, fx, dϕ_0)
        return α, fx
    catch y
        # throw(y)
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end

end


function HagerZhangLineSearch(
    iter::IterationType,
    gx, fx,
    x::Tx,
    s::Tg;
    α₀::R=1.0
) where {IterationType,Tx,Tg,R<:Real}

    ϕ(α) = iter.f(x .+ α .* s)
    function dϕ(α)
        if iter.g !== nothing
            gv = iter.g(x + α .* s)
        else
            gv = similar(s)
            iter.ga(gv, x + α .* s)
        end
        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = iter.f(x .+ α .* s)
        if iter.g !== nothing
            gv = iter.g(x + α .* s)
        else
            gv = similar(s)
            iter.ga(gv, x + α .* s)
        end
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)

    try
        α, fx, it = lsa(ϕ, dϕ, ϕdϕ, α₀, fx, dϕ_0)
        return α, fx, it
    catch y
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end
end

function HagerZhangLineSearch(
    f, g,
    gx, fx,
    x::Tx,
    s::Tg;
    α₀::R=1.0
) where {Tx,Tg,R<:Real}

    ϕ(α) = f(x .+ α .* s)
    function dϕ(α)

        gv = g(x + α .* s)

        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = f(x .+ α .* s)
        gv = g(x + α .* s)
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)
    try
        α, fx, it = lsa(ϕ, dϕ, ϕdϕ, α₀, fx, dϕ_0)
        return α, fx, it
    catch y
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end

end