import Pkg


Pkg.rm("DRSOM")
Pkg.develop(path=".")
Pkg.instantiate()