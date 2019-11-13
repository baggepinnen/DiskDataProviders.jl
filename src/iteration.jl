read_and_transform(d::AbstractDiskDataProvider, fileindex) = d.transform === nothing ? deserialize(d.files[fileindex]) : d.transform(deserialize(d.files[fileindex]))

# read_and_transform(d::AbstractDiskDataProvider{<:Any, Nothing}, fileindex) = d.transform === nothing ? (deserialize(d.files[fileindex]), nothing) : (d.transform(deserialize(d.files[fileindex])), nothing)

copyto_batch!(dst::AbstractArray{T,4},src,i) where T = dst[:,:,:,i] .= src
copyto_batch!(dst::Matrix,src,i)  = dst[:,i] .= src

function buffered_batch(d::QueueDiskDataProvider{<:Any, YT}, inds) where YT
    for (i,j) in enumerate(inds)
        x,y = d.queue[j]
        copyto_batch!(d.x_batch, x, i)
        YT === Nothing || (d.y_batch[i] = y)
    end
    (d.x_batch, d.y_batch)
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
    (d.x_batch, d.y_batch)
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
    struct BufferedIterator{T <: AbstractDiskDataProvider}

Creates an iterator which uses the underlying buffer in the dataset.
"""
struct BufferedIterator{T <: AbstractDiskDataProvider}
    d::T
end

"""
    struct UnbufferedIterator{T <: AbstractDiskDataProvider}

Creates an iterator which does not use the underlying buffer in the dataset.
"""
struct UnbufferedIterator{T <: AbstractDiskDataProvider}
    d::T
end

Base.show(io::IOContext, m::MIME{Symbol("text/plain")}, bw::BatchView{<:Any,<:BufferedIterator}) = show(io, m, "BatchView{BufferedIterator} of length $(length(bw))")
Base.show(io::IOContext, m::MIME{Symbol("text/plain")}, bw::BatchView{<:Any,<:UnbufferedIterator}) = show(io, m, "BatchView{UnbufferedIterator} of length $(length(bw))")


function Base.iterate(d::BufferedIterator{<: QueueDiskDataProvider}, state=0)
    state == length(d) && return nothing
    (sample_random_datapoint(d),state+1)
end

function Base.iterate(d::BufferedIterator{<: ChannelDiskDataProvider}, state=0)
    state == length(d) && return nothing
    (take!(d.d),state+1)
end

function Base.iterate(d::UnbufferedIterator, state=1)
    state > length(d.d) && (return nothing)
    d.d[state], state+1
end

function Base.iterate(d::AbstractDiskDataProvider, args...)
    iterate(UnbufferedIterator(d), args...)
end

# function Base.iterate(ubw::UnbufferedIterator, state)
#     res = iterate(ubw.inner, state)
#     res === nothing && return nothing
#     unbuffered_batch(ubw.d.d, res[1]), res[2]
# end

function Base.iterate(ds::DataSubset{<:AbstractDiskDataProvider}, state=nothing)
    inds = ds.indices
    d = ds.data
    buffered_batch(d,inds)
end


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
    yt,yv = stratifiedobs(d.labels, p, args...; kwargs...)
    split(d, first(yt.indices), first(yv.indices))
end


Base.length(d::AbstractDiskDataProvider) = d.length
Base.length(d::BufferedIterator) = d.d.length
Base.length(d::UnbufferedIterator) = length(d.d)


# Base.pairs(d::BufferedIterator) = enumerate(d)
Base.pairs(d::UnbufferedIterator) = enumerate(d.d)
Base.pairs(d::AbstractDiskDataProvider) = enumerate(d)
Base.getindex(d::AbstractDiskDataProvider, i) = read_and_transform(d,i)

"""
    batchview(d::AbstractDiskDataProvider; size=d.batchsize, kwargs...)

Create a batch iterator that iterates batches with the batch size defined at the creation of the DiskDataProvider.
"""
MLDataUtils.batchview(d; size, kwargs...)

function MLDataUtils.batchview(d::AbstractDiskDataProvider; size=d.batchsize, kwargs...)
    isready(d) || error("You can only create a buffered iterator after you have started reading elements into the buffer.")
    batchview(BufferedIterator(d); size=size, kwargs...)
end
MLDataUtils.batchview(d::UnbufferedIterator; size=d.batchsize, kwargs...) = batchview(UnbufferedIterator(d); size=size, kwargs...)

"""
    LearnBase.nobs(d)
    
Get the number of observations in the dataset
"""
LearnBase.nobs(d)

LearnBase.nobs(d::QueueDiskDataProvider)  = length(d)
LearnBase.nobs(d::ChannelDiskDataProvider)  = length(d)

LearnBase.nobs(d::BufferedIterator) = length(d.d)
LearnBase.getobs(d::BufferedIterator, inds) = buffered_batch(d.d,inds)
LearnBase.datasubset(d::BufferedIterator, inds, ::ObsDim.Undefined) = buffered_batch(d.d,inds)

LearnBase.nobs(d::UnbufferedIterator) = length(d.d)
LearnBase.getobs(d::UnbufferedIterator, inds) = unbuffered_batch(d.d,inds)
LearnBase.datasubset(d::UnbufferedIterator, inds, ::ObsDim.Undefined) = unbuffered_batch(d.d,inds)


Base.show(io::IO, d::BufferedIterator) = println(io, "$(typeof(d)), length: $(length(d.d))")
Base.show(io::IO, d::UnbufferedIterator) = println(io, "$(typeof(d)), length: $(length(d.d))")
