
[![Build Status](https://travis-ci.org/baggepinnen/DiskDataProviders.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DiskDataProviders.jl)
[![codecov](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl)

# DiskDataProviders

```@setup memory
using DiskDataProviders, Test, Serialization
```

This package implements datastructures that are iterable and backed by a buffer that is fed by data from disk. If Reading and preproccesing data is faster than one training step, it's recommended to use a [`ChannelDiskDataProvider`](@ref), if the training step is fast but reading data takes long time, [`QueueDiskDataProvider`](@ref) is recommended. Both types do the reading on a separate thread, so make sure Julia is started with at least two threads.

My personal use case for this package is training convolutional DL models using Flux. This package does not take care of the transfer of data to the GPU, as I have not managed to do this on a separate thread.

## Supervised vs unsupervised
If the task is supervised, you may supply labels using the keyword `labels`, see example below. If the dataset has labels, it iterates tuples `(x,y)`. If no labels are supplied, it iterates only inputs `x`. To create an unsupervised dataset with no labels, use `Nothing` as the label type, e.g. `DiskDataProvider{xType, Nothing}`.

## Usage example
```@example memory
using DiskDataProviders, Test, Serialization

# === Create some random example data ===
dirpath = mktempdir()*"/"
N = 100
T = 500
batch_size = 2
queue_length = 5 # Length of the internal buffer, it's a good idea to make this be some integer multiple of the batch size.
labs = rand(1:5, N)
for i = 1:N
    a = randn(T)
    serialize(dirpath*"$(i).bin", (a, labs[i]))
end

files = dirpath .* string.(1:N) .* ".bin"

# === Create a DiskDataProvider ===
dataset = ChannelDiskDataProvider{Vector{Float64}, Int}((T,), batch_size, queue_length; labels=labs, files=files)
```

The dataset is iterable and can be used in loops etc. One can also create a [`batchview`](@ref DiskDataProviders.batchview), which is an iterator over batches. The batch size is defined when the DiskDataProvider is created.

```@repl memory
# === Example usage of the provider ===
datasett, datasetv = stratifiedobs(dataset, 0.75)

sort(dataset.ulabels) == 1:5

x,y = first(dataset) # Get one datapoint

t = start_reading(dataset) # this function initiates the reading into the buffer

wait(dataset) # Wait for the reading to start before proceeding

bw = batchview(dataset)

xb,yb = first(bw) # Get the first batch from the buffer

for (x,y) in bw # Iterate the batches in the batchview
    # do something with the data
end

stop!(dataset) # Stop reading into the buffer
```

If your data has more dimensions than 1, e.g., inputs are matrices or 3d-tensors, you create a DiskDataProvider like this
```julia
dataset = ChannelDiskDataProvider((nrows,ncols,nchannels), batchsize, queuelength; labels=labs, files=files)
```
notice that you have to provide `nchannels`, which is `1` if the input is a matrix.


# Preprocess data
All functionality in this package operates on serialized, preprocessed data files. Serialized files are fast to read, and storing already preprocessed data cuts down on overhead. This package does currently not support arbitrary file formats. The files are read using Julias built in deserializer.


# Iterators
- If you simply iterate over an `AbstractDiskDataProvider`, you will iterate over each datapoint in the sequence determined by the vector of file paths. This iteration is buffered by a buffer unique to the iterator.
- [`batchview`](@ref DiskDataProviders.batchview) creates a buffered iterator over batches.
- [`unbuffered`](@ref) an iterator that is not buffered.
- [`buffered`](@ref) iterates over single datapoints from the buffer.
- [`full_batch`](@ref) creates one enormous batch of the entire dataset.
- [`unbuffered_batchview`](@ref) Iterates over batches, unbuffered.
- For unsupervised datasets (without labels), the buffers are populated by randomly permuting the data files (shuffling). Using the default file iterator, all datapoints are visited in the same order in each epoch.
- For supervised datasets, unique labels are cycled through and a datapoint with that label is drawn uniformly at random.

Typically, you want to use [`batchview`](@ref DiskDataProviders.batchview) for training. If you have a small enough dataset (e.g. for validation), you may want to use [`full_batch`](@ref), especially if this fits into the GPU memory. Batches are structured according to Flux's notion of a batch, e.g., the last dimension is the batch dimension.

# Exported functions and types
## Index

```@index
```
```@autodocs
Modules = [DiskDataProviders, MLDataPattern]
Pages = ["DiskDataProviders.jl", "iteration.jl"]
Private = false
```
