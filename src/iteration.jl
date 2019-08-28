read_and_transform(d::AbstractDiskDataProvider, fileindex) = d.transform === nothing ? deserialize(d.files[fileindex]) : d.transform(deserialize(d.files[fileindex]))

function buffered_batch(d::QueueDiskDataProvider, inds)
    for (i,j) in enumerate(inds)
        x,y = d.queue[j]
        d.x_batch[:,:,:,i] .= x
        d.y_batch[i] = y
    end
    (d.x_batch, d.y_batch)
end

# inds are ignored for this iterator
function buffered_batch(d::ChannelDiskDataProvider, inds)
    # isready(d.channel) || error("There are no elements in the channel. Either start reading or switch to a QueueDiskDataProvider")
    for (i,j) in enumerate(inds)
        x,y = take!(d)
        d.x_batch[:,:,:,i] .= x
        d.y_batch[i] = y
    end
    (d.x_batch, d.y_batch)
end

function unbuffered_batch(d::AbstractDiskDataProvider{XT,YT}, inds)::Tuple{Array{eltype(XT),4},Vector{YT}} where {XT,YT}
    mi,ma = extrema(inds)
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(inds))
    Y = similar(d.y_batch, length(inds))
    for (i,j) in enumerate(inds)
        x,y = read_and_transform(d,j)
        X[:,:,:,i] .= x
        Y[i] = y
    end
    X,Y
end

function full_batch(d::AbstractDiskDataProvider{XT,YT}) where {XT,YT}
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(d))
    Y = similar(d.y_batch, length(d))
    for i = 1:length(d)
        x,y = read_and_transform(d,i)
        X[:,:,:,i] .= x
        Y[i] = y
    end
    X,Y
end

struct BufferedIterator{T <: AbstractDiskDataProvider}
    d::T
end

struct UnbufferedIterator{T <: AbstractDiskDataProvider}
    d::T
end

Base.show(io::IOContext, m::MIME, bw::BatchView) = show(io, m, "BatchView{BufferedIterator}")
Base.show(io::IOContext, m::MIME, bw::BatchView) = show(io, m, "BatchView{UnbufferedIterator}")


function Base.iterate(d::BufferedIterator{<: QueueDiskDataProvider}, state=0)
    state == length(d) && return nothing
    (sample_random_datapoint(d),state+1)
end

function Base.iterate(d::BufferedIterator{<: ChannelDiskDataProvider}, state=0)
    state == length(d) && return nothing
    (take!(d),state+1)
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

function MLDataUtils.batchview(d::AbstractDiskDataProvider; size=d.batchsize, kwargs...)
    isready(d) || error("You can only create a buffered iterator after you have started reading elements into the buffer.")
    batchview(BufferedIterator(d); size=size, kwargs...)
end
MLDataUtils.batchview(d::UnbufferedIterator; size=d.batchsize, kwargs...) = batchview(UnbufferedIterator(d); size=size, kwargs...)

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
