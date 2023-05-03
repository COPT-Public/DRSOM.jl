ENV["GKSwstype"] = "100"
using ADNLPModels
using Documenter
using Printf
using AdaptiveRegularization

pages = [
  "Introduction" => "index.md",
  "Tutorial" => "benchmark.md",
  "Do it yourself" => "doityourself.md",
  "Reference" => "reference.md",
]

makedocs(
  sitename = "AdaptiveRegularization.jl",
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [AdaptiveRegularization],
  pages = pages,
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/AdaptiveRegularization.jl.git",
  push_preview = true,
  devbranch = "main",
)
