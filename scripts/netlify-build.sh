#!/usr/bin/env bash
set -euo pipefail
set -x

# Configure Julia version (override by setting JULIA_VERSION env in Netlify)
JULIA_VERSION="${JULIA_VERSION:-1.8.5}"
JULIA_MAJOR_MINOR="${JULIA_VERSION%.*}"
ARCH="x86_64"
TARBALL="julia-${JULIA_VERSION}-linux-${ARCH}.tar.gz"
BASE_URL="https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MAJOR_MINOR}"

curl -fsSL "${BASE_URL}/${TARBALL}" -o "${TARBALL}"
tar -xzf "${TARBALL}"
export PATH="$PWD/julia-${JULIA_VERSION}/bin:$PATH"
julia --version

# Build docs
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("docs/make.jl")'


