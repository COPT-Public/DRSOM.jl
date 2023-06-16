import Pkg

try
    Pkg.rm("DRSOM")
catch 
end
try
    Pkg.rm("AdaptiveRegularization")
catch
end
try
    Pkg.rm("LIBSVMFileIO")
catch
end
Pkg.develop(path=".")
Pkg.develop(path="test/third-party/AdaptiveRegularization.jl")
Pkg.develop(path="test/third-party/LIBSVMFileIO.jl")
Pkg.instantiate()