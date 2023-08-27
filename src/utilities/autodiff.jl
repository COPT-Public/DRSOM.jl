using ForwardDiff
using ReverseDiff

# not recommended
function hessian(f, x)
    h = ForwardDiff.hessian(f, x)
    return h
end

#########################################
# Hessian vector produce
#########################################
# use forward mode
# exact mode: not recommended
function hessf(f, state)
    d = state.d
    x = state.x
    function gd(x)
        a = ForwardDiff.gradient(f, x)
        return a' * d
    end
    function gs(x)
        a = ForwardDiff.gradient(f, x)
        return a' * a / 2
    end
    Hg = ForwardDiff.gradient(gs, x)
    Hd = ForwardDiff.gradient(gd, x)
    return Hg, Hd
end

function hessf(f, d, x)
    function gd(x)
        a = ForwardDiff.gradient(f, x)
        return a' * d
    end
    function gs(x)
        a = ForwardDiff.gradient(f, x)
        return a' * a / 2
    end
    Hg = ForwardDiff.gradient(gs, x)
    Hd = ForwardDiff.gradient(gd, x)
    return Hg, Hd
end

# approximation
function hessfa(f, x, v, hvp, ∇hvp, ∇f; scale::Real=200.0, cfg::ForwardDiff.GradientConfig)
    ForwardDiff.gradient!(∇hvp, f, x + v ./ scale, cfg)
    copy!(hvp, scale * (∇hvp - ∇f))
end


# use backward mode

# approximation
function hessba(x, v, hvp, ∇hvp, ∇f; scale::Real=200.0, tp::ReverseDiff.CompiledTape)
    # Hessian-vector finite diff
    ReverseDiff.gradient!(∇hvp, tp, x + v ./ scale)
    copy!(hvp, scale * (∇hvp - ∇f))
end
