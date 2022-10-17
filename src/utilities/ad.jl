using ForwardDiff
using ReverseDiff

function gradientf(f, x)
    fx = ForwardDiff.gradient(f, x)
    return fx
end

# it is not advised to do so
function hessian(f, x)
    h = ForwardDiff.hessian(f, x)
    return h
end
##############################
# compute Hg and Hd
##############################
# use forward mode
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
function hessfa(f, g_buffer, d, x; scale::Real=200.0, cfg::ForwardDiff.GradientConfig)
    ForwardDiff.gradient!(g_buffer, f, x, cfg)
    Hg = scale * (ForwardDiff.gradient(f, x + g_buffer ./ scale, cfg) - g_buffer)
    Hd = scale * (ForwardDiff.gradient(f, x + d ./ scale, cfg) - g_buffer)
    return Hg, Hd
end

function hessfa(f, state; scale::Real=200.0, cfg::ForwardDiff.GradientConfig)
    x = state.x
    d = state.d
    ForwardDiff.gradient!(state.∇f, f, x, cfg)
    Hg = scale * (ForwardDiff.gradient(f, x + state.∇f ./ scale, cfg) - state.∇f)
    Hd = scale * (ForwardDiff.gradient(f, x + d ./ scale, cfg) - state.∇f)
    return Hg, Hd
end


# use backward mode

# approximation
function hessba(g_buffer, g_temp_buffer, d, x; scale::Real=200.0, tp::ReverseDiff.CompiledTape)
    ReverseDiff.gradient!(g_buffer, tp, x)
    # Hg
    ReverseDiff.gradient!(g_temp_buffer, tp, x + g_buffer ./ scale)
    Hg = scale * (g_temp_buffer - g_buffer)
    # Hd
    ReverseDiff.gradient!(g_temp_buffer, tp, x + d ./ scale)
    Hd = scale * (g_temp_buffer - g_buffer)
    return Hg, Hd
end

function hessba(state; scale::Real=200.0, tp::ReverseDiff.CompiledTape)
    x = state.x
    d = state.d
    ReverseDiff.gradient!(state.∇f, tp, x)
    # Hg
    ReverseDiff.gradient!(state.∇fb, tp, x + state.∇f ./ scale)
    Hg = scale * (state.∇fb - state.∇f)
    # Hd
    ReverseDiff.gradient!(state.∇fb, tp, x + d ./ scale)
    Hd = scale * (state.∇fb - state.∇f)
    return Hg, Hd
end

function hessba(state, v; scale::Real=200.0, tp::ReverseDiff.CompiledTape)
    x = state.x
    # Hessian-vector finite diff
    ReverseDiff.gradient!(state.∇fb, tp, x + v ./ scale)
    Hv = scale * (state.∇fb - state.∇f)
    Hv
end
