
mutable struct Counting{TF}
    f::TF
    counter::Int
end

Counting(f) = Counting(f, 0)

function (c::Counting)(args...)
    c.counter += 1
    return c.f(args...)
end
