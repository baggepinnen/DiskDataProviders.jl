
[![Build Status](https://travis-ci.org/baggepinnen/DiskDataProviders.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DiskDataProviders.jl)
[![codecov](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl)

# DiskDataProviders

```@setup memory
using DiskDataProviders, Test, Serialization, MLDataUtils
```

This package implements datastructures that are iterable and backed by a buffer that is fed by data from disc. If Reading and preproccesing data is faster than one training step, it's recommended to use a [`ChannelDiskDataProvider`](@ref), if the training step is fast but reading data takes long time, [`QueueDiskDataProvider`](@ref) is recommended. Both types do the reading on a separate thread, so make sure Julia is started with at least two threads.

Usage example
```@example memory
using DiskDataProviders, Test, Serialization, MLDataUtils

# === Create some random example data ===
dirpath = mktempdir()*"/"
N = 100
T = 500
batch_size = 2
queue_length = 5 # Length of the internal buffer.
labs = rand(1:5, N)
for i = 1:N
    a = randn(T)
    serialize(dirpath*"$(i).bin", (a, labs[i]))
end

files = dirpath .* string.(1:N) .* ".bin"

# === Create a DiskDataProvider ===
dataset = ChannelDiskDataProvider{Vector{Float64}, Int}((T,), batch_size, queue_length; labels=labs, files=files)
```

The dataset is iterable and can be used in loops etc. One can also create a [`BatchView`](@ref), which is an iterator over batches. The batch size is defined when the DiskDataProvider is created.

```@repl memory
# === Example usage of the provider ===
datasett, datasetv = stratifiedobs(dataset, 0.75)

sort(dataset.ulabels) == 1:5

x,y = first(dataset) # Get one datapoint

t = start_reading(dataset) # this function initiates the reading into the buffer

wait(dataset) # Wait for the reading to start before proceeding

bw = batchview(dataset);

xb,yb = first(bw) # Get the first batch from the buffer

for (x,y) in bw # Iterate the batches in the batchview
    # do something with the data
end

stop!(dataset) # Stop reading into the buffer
```

# Exported functions and types
## Index

```@index
```
```@autodocs
Modules = [DiskDataProviders]
Private = false
```
