[![Build Status](https://travis-ci.org/baggepinnen/DiskDataProviders.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DiskDataProviders.jl)
[![codecov](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl)


<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/stable) -->
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/latest)

# DiskDataProviders.jl

Usage example
```julia
using DiskDataProviders, MLDataUtils

files = # vector of strings to serialized files
labs = fill(nothing, length(files))

transform(x) = # some pre transformation of x

dataset = ChannelDiskDataProvider{Matrix{Float32}, Nothing}(data_size_tuple, 2, channel_length, labels=labs, files=files, transform=transform)

t = start_reading(dataset) # Start reading of the data
istaskstarted(t) && !istaskfailed(t) && wait(datasett)
bw =  batchview(dataset)
x,y = first(bw) # The first data point
```
