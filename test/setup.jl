import Pkg

Pkg.develop(path=".")
Pkg.add(path="test/third-party/AdaptiveRegularization.jl")
Pkg.add(path="test/third-party/LIBSVMFileIO.jl")