__precompile__()
module DRSOM

using Printf

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

# various utilities
include("utilities/logger.jl")
include("utilities/ad.jl")
include("utilities/fb_tools.jl")
include("utilities/iteration_tools.jl")
include("utilities/display_tools.jl")
include("utilities/trs.jl")
include("utilities/counter.jl")
include("utilities/interpolation.jl")
include("utilities/adaptive.jl")
include("utilities/lanczos.jl")


# algorithm implementations

include("algorithms/interface.jl")
include("algorithms/legacy/drsom_legacy.jl")
include("algorithms/drsom.jl")
include("algorithms/hsodm.jl")
include("algorithms/pfh.jl")
include("algorithms/utr.jl")

# Algorithm Aliases
DRSOM2 = DimensionReducedSecondOrderMethod
HSODM = HomogeneousSecondOrderDescentMethod
PFH = PathFollowingHSODM
UTR = UniversalTrustRegion

function __init__()
    Logger.initialize()
end

export Result
export DRSOM2, HSODM, PFH, UTR
end # module
