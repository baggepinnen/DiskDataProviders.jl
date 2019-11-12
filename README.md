[![Build Status](https://travis-ci.org/baggepinnen/DiskDataProviders.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DiskDataProviders.jl)
[![codecov](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl)


<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/stable) -->
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/latest)

# DiskDataProviders.jl

Usage example
```julia
using DiskDataProviders, Test, Serialization, MLDataUtils

# === Create some random example data ===
dirpath = mktempdir()*"/"
N = 100
T = 500
bs = 2
labs = rand(1:5, N)
for i = 1:N
    a = randn(T)
    serialize(dirpath*"$(i).bin", (a, labs[i]))
end

files = dirpath .* string.(1:N) .* ".bin"

# === Create a DiskDataProvider ===
dataset = ChannelDiskDataProvider{Vector{Float64}, Int}((T,), bs, 5; labels=labs, files=files)


# === Example usage of the provider ===
datasett, datasetv = stratifiedobs(dataset, 0.75)

sort(dataset.ulabels) == 1:5

x,y = first(dataset) # Get one datapoint

t = start_reading(dataset) # this function initiates the reading into the buffer

bw = batchview(dataset)

xb,yb = first(bw) # Get the first batch from the buffer

for (x,y) in bw # Iterate the batches in the batchview
    ...
end

stop!(dataset) # Stop reading into the buffer
```
