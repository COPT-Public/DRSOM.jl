# LIBSVMFileIO

[![Build Status](https://travis-ci.org/mvmorin/LIBSVMFileIO.jl.svg?branch=master)](https://travis-ci.org/mvmorin/LIBSVMFileIO.jl)
[![codecov](https://codecov.io/gh/mvmorin/LIBSVMFileIO.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mvmorin/LIBSVMFileIO.jl)

Provides functions for reading and writing data files in the format used by
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).


## Introduction
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) has collected a number of
[data sets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) in a
common format for benchmarking purposes. Easy access to these publicly data sets
are useful even if one is not interested in using LIBSVM specifically. This a
bare-bones, stand-alone package whose only purpose is to read and write data to
files in the LIBSVM-format.

The package supports reading both classification and regression data as well as
multi-label data. The default behaviour is to load each data point as a sparse
vector. Partial loading where only a subset of the data points are loaded is
also supported.

## Usage
The package is currently not registered but can be installed as any other package
by hitting ']' in the Julia REPL and running
```julia
pgk> add https://github.com/mvmorin/LIBSVMFileIO.jl
```

After installation the package can be loaded like any other
```julia
julia> using LIBSVMFileIO
```

The package provides three functions `libsvmread`, `libsvmwrite`, and
`libsvmsize`. Documentation is provided via the built-in help functionality. Run
any of
```julia
julia> ?libsvmread

julia> ?libsvmwrite

julia> ?libsvmsize
```
for more information.
