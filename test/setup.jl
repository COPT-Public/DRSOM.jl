import Pkg

Pkg.develop(path=".")
Pkg.develop(path="test/third-party/AdaptiveRegularization.jl")
Pkg.develop(path="test/third-party/LIBSVMFileIO.jl")
Pkg.instantiate()