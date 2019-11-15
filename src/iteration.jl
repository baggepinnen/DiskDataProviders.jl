read_and_transform(d::AbstractDiskDataProvider, fileindex) = d.transform === nothing ? deserialize(d.files[fileindex]) : d.transform(deserialize(d.files[fileindex]))

# read_and_transform(d::AbstractDiskDataProvider{<:Any, Nothing}, fileindex) = d.transform === nothing ? (deserialize(d.files[fileindex]), nothing) : (d.transform(deserialize(d.files[fileindex])), nothing)

copyto_batch!(dst::AbstractArray{T,4},src,i) where T = dst[:,:,:,i] .= src
copyto_batch!(dst::Matrix,src,i)  = dst[:,i] .= src



function buffered_batch(d::QueueDiskDataProvider{<:Any, YT}, inds) where YT
    for (i,j) in enumerate(inds)
        x,y = d.queue[(j-1)%queuelength(d) + 1]
        copyto_batch!(d.x_batch, x, i)
        YT === Nothing || (d.y_batch[i] = y)
    end
    YT === Nothing ? d.x_batch : (d.x_batch, d.y_batch)
end

# inds are ignored for this iterator
function buffered_batch(d::ChannelDiskDataProvider{<:Any, YT}, inds) where YT
    # isready(d.channel) || error("There are no elements in the channel. Either start reading or switch to a QueueDiskDataProvider")
    for (i,j) in enumerate(inds)
        x,y = take!(d)
        size(d.x_batch,1) == size(x,1) || continue
        copyto_batch!(d.x_batch, x, i)
        YT === Nothing || (d.y_batch[i] = y)
    end
    YT === Nothing ? d.x_batch : (d.x_batch, d.y_batch)
end

function unbuffered_batch(d::AbstractDiskDataProvider{XT,YT}, inds) where {XT,YT}
    mi,ma = extrema(inds)
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(inds))
    Y = similar(d.y_batch, length(inds))
    for (i,j) in enumerate(inds)
        x,y = read_and_transform(d,j)
        copyto_batch!(X, x, i)
        Y[i] = y
    end
    if !(XT <: AbstractVector)
        X = convert(Array{eltype(XT),4}, X)
    end
    X,Y
end

"""
    full_batch(d::AbstractDiskDataProvider)

Returns a matrix with the entire dataset.
"""
full_batch(d::AbstractDiskDataProvider)
function full_batch(d::AbstractDiskDataProvider{XT,YT}) where {XT,YT}
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(d))
    Y = similar(d.y_batch, length(d))
    for i = 1:length(d)
        x,y = read_and_transform(d,i)
        copyto_batch!(X, x, i)
        Y[i] = y
    end
    X,Y
end

function unbuffered_batch(d::AbstractDiskDataProvider{XT,Nothing}, inds) where {XT}
    mi,ma = extrema(inds)
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(inds))
    for (i,j) in enumerate(inds)
        x,y = read_and_transform(d,j)
        copyto_batch!(X, x, i)
    end
    if !(XT <: AbstractVector)
        X = convert(Array{eltype(XT),4}, X)
    end
    X
end

function full_batch(d::AbstractDiskDataProvider{XT,Nothing}) where {XT}
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(d))
    for i = 1:length(d)
        x,y = read_and_transform(d,i)
        copyto_batch!(X, x, i)
    end
    X
end

"""
    buffered(d::AbstractDiskDataProvider)

Creates an iterator which uses the underlying buffer in the dataset.
"""
buffered(d::AbstractDiskDataProvider)
@resumable function buffered(d::QueueDiskDataProvider)
    for state in 1:length(d)
        @yield sample_random_datapoint(d)
    end
end

@resumable function buffered(d::ChannelDiskDataProvider)
    for state in 1:length(d)
        @yield take!(d)
    end
end

"""
    unbuffered(d::AbstractDiskDataProvider)

Creates an iterator which does not use the underlying buffer in the dataset.
"""
unbuffered(d::AbstractDiskDataProvider)
@resumable function unbuffered(d::AbstractDiskDataProvider)
    for i in 1:length(d)
        @yield d[i]
    end
end

"""
    batchview(d::AbstractDiskDataProvider, size=d.batchsize; kwargs...)

Create a batch iterator that iterates batches with the batch size defined at the creation of the DiskDataProvider.
"""
batchview(d::AbstractDiskDataProvider)

@resumable function batchview(d::AbstractDiskDataProvider, size=d.batchsize)
    isready(d) || error("You can only create a buffered iterator after you have started reading elements into the buffer.")
    for inds in Iterators.partition(1:length(d), size)
        @yield buffered_batch(d, inds)
    end
end
"""
    unbuffered_batchview(d::AbstractDiskDataProvider, size=d.batchsize)

Iterate unbuffered batches. See also [`batchview`](@ref)
"""
unbuffered_batchview(d::AbstractDiskDataProvider, size=d.batchsize)

@resumable function unbuffered_batchview(d::AbstractDiskDataProvider, size=d.batchsize)
    for inds in Iterators.partition(1:length(d), size)
        @yield unbuffered_batch(d, inds)
    end
end


function Base.iterate(d::AbstractDiskDataProvider)
    d1 = d[1]
    ch = Channel{typeof(d1)}(2, spawn=true) do ch
        foreach(i->put!(ch, d[i]), 2:length(d))
    end
    return d1, (2,ch)
end

function Base.iterate(d::AbstractDiskDataProvider, state)
    i,ch = state
    (i > length(d) || !isopen(ch)) && return nothing
    return take!(ch), (i+1, ch)
end

Base.eltype(d::AbstractDiskDataProvider{XT,YT}) where {XT,YT} = Tuple{XT,YT}
Base.eltype(d::AbstractDiskDataProvider{XT,Nothing}) where {XT} = XT


"""
    stratifiedobs(d::AbstractDiskDataProvider, p::AbstractFloat, args...; kwargs...)

Partition the data into multiple disjoint subsets proportional to the
value(s) of p. The observations are assignmed to a data subset using
stratified sampling without replacement. These subsets are then returned
as a Tuple of subsets, where the first element contains the fraction of
observations of data that is specified by the first float in p.

 For example, if p is a Float64 itself, then the return-value will be a
tuple with two elements (i.e. subsets), in which the first element
contains the fraction of observations specified by p and the second
element contains the rest. In the following code the first subset train
will contain around 70% of the observations and the second subset test
the rest. The key difference to splitobs is that the class distribution
in y will actively be preserved in train and test.

 `train, test = stratifiedobs(diskdataprovider, 0.7)`
"""
function MLDataUtils.stratifiedobs(d::AbstractDiskDataProvider, p::AbstractFloat, args...; kwargs...)
    yt,yv = MLDataUtils.stratifiedobs(d.labels, p, args...; kwargs...)
    split(d, first(yt.indices), first(yv.indices))
end


Base.length(d::AbstractDiskDataProvider) = d.length
Base.pairs(d::AbstractDiskDataProvider) = enumerate(d)
Base.getindex(d::AbstractDiskDataProvider, i) = read_and_transform(d,i)


# Base.length(d::ResumableFunctions.FiniteStateMachineIterator) = length(d.d) # This is type piracy
