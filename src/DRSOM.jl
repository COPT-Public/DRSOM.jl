__precompile__()
module DRSOM

using Printf

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

# various utilities
# include("utilities/logger.jl")
include("utilities/autodiff.jl")
include("utilities/fbtools.jl")
include("utilities/iterationtools.jl")
include("utilities/displaytools.jl")
include("utilities/counter.jl")
include("utilities/interpolation.jl")
include("utilities/linesearches.jl")
include("utilities/homogeneous.jl")
include("utilities/trustregion.jl")
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
    # Logger.initialize()
end

export Result
export DRSOM2, HSODM, PFH, UTR
end # module
