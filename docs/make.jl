using Documenter, DiskDataProviders
using DiskDataProviders, LearnBase, MLDataUtils, MLDataPattern

makedocs(
    sitename = "DiskDataProviders",
    # format = LaTeX(),
    format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
    modules = [DiskDataProviders, MLDataUtils, MLDataPattern, LearnBase],
    pages = ["index.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/baggepinnen/DiskDataProviders.jl"
)
