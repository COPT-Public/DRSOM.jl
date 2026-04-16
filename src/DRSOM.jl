__precompile__()
module DRSOM

using Printf

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

# various utilities
include("utilities/autodiff.jl")
include("utilities/fbtools.jl")
include("utilities/iterationtools.jl")
include("utilities/displaytools.jl")
include("utilities/counter.jl")
include("utilities/interpolation.jl")
include("utilities/linesearches.jl")

# subproblem solvers
include("utilities/subp/atrs/ATRS.jl")
include("utilities/subp/ghm.jl")
include("utilities/subp/trs.jl")
include("utilities/subp/lanczos.jl")
include("utilities/subp/cubic.jl")

# algorithm implementations
include("algorithms/interface.jl")
include("algorithms/legacy/drsom_legacy.jl")
include("algorithms/drsom.jl")
include("algorithms/drsomex.jl")
include("algorithms/hsodm.jl")
include("algorithms/pfh.jl")
include("algorithms/utr.jl")
include("algorithms/atr.jl")
include("algorithms/atrms.jl")
include("algorithms/hacubic.jl")

# my implementation of other algorithms
include("others/cubicreg_vanilla.jl")

# nonsmooth algorithms
include("nonsmooth/fbe.jl")
include("nonsmooth/fbedrsom.jl")

# Algorithm Aliases
DRSOM2 = DimensionReducedSecondOrderMethod
DRSOMEx = DimensionReducedSecondOrderMethodEx
HSODM = HomogeneousSecondOrderDescentMethod
PFH = PathFollowingHSODM
FBEDRSOM = ForwardBackwardDimensionReducedSecondOrderMethod
UTR = UniversalTrustRegion
ATR = AcceleratedUniversalTrustRegion
ATRMS = AcceleratedUniversalTrustRegionMonteiroSvaiter
HaCubic = HistoryAwareCubicRegularization
function __init__()
    # Logger.initialize()
end

export Result
export DRSOM2, DRSOMEx
export FBEDRSOM
export HSODM, PFH
export UTR, ATR, ATRMS
export HaCubic

# other algorithms
export CubicRegularizationVanilla
end # module
