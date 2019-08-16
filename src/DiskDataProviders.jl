module DiskDataProviders
using MLDataUtils, LearnBase, DataFrames, Dates, Serialization

import Base.Threads: nthreads, threadid, @spawn, SpinLock

export DiskDataProvider, label2filedict, start_reading, stop!, UnbufferedBatchView, labels

Serialization.serialize(filename::AbstractString, data) = open(f->serialize(f, data), filename, "w")
Serialization.deserialize(filename) = open(f->deserialize(f), filename)

Base.@kwdef mutable struct DiskDataProvider{XT,YT}
    batchsize  ::Int           = 8
    labels     ::Vector{YT}
    files      ::Vector{String}
    length     ::Int           = length(labels)
    ulabels    ::Vector{YT}    = unique(labels)
    label2files::Dict{YT,Vector{String}} = label2filedict(labels, files)
    queue_full ::Threads.Event = Threads.Event()
    queue      ::Vector{Tuple{XT,YT}}
    label_iterator             = Iterators.cycle(unique(labels))
    label_iterator_state       = nothing
    position   ::Int           = 1
    reading    ::Bool          = false
    queuelock  ::SpinLock      = SpinLock()
    x_batch    ::Array{Float32,4}
    y_batch    ::Vector{YT}
end

function DiskDataProvider{XT,YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT,YT}
    queue   = Vector{Tuple{XT,YT}}(undef, queuelength)
    x_batch = Array{Float32,4}(undef, xsize..., batchsize)
    y_batch = Vector{YT}(undef, batchsize)
    DiskDataProvider{XT,YT}(;
        queue   = queue,
        batchsize = batchsize,
        x_batch = x_batch,
        y_batch = y_batch,
        kwargs...)
end

function DiskDataProvider(d::DiskDataProvider, inds::AbstractArray)
    DiskDataProvider(
        batchsize            = d.batchsize,
        length               = length(inds),
        labels               = d.labels[inds],
        files                = d.files[inds],
        ulabels              = d.ulabels,
        queue_full           = Threads.Event(),
        queue                = similar(d.queue),
        label_iterator       = d.label_iterator,
        label_iterator_state = nothing,
        position             = 1,
        reading              = false,
        queuelock            = SpinLock(),
        x_batch              = similar(d.x_batch),
        y_batch              = similar(d.y_batch)
    )
end

function Base.split(d::DiskDataProvider,i1,i2)
    DiskDataProvider(d,i1), DiskDataProvider(d,i2)
end

Base.show(io::IO, d::DiskDataProvider) = println(io, "$(typeof(d)), length: $(length(d))")

labels(d::DiskDataProvider)    = map(findfirst, eachrow(d.labels .== reshape(d.ulabels,1,:)))
labels(d::Vector{<:Tuple})     = last.(d)
nclasses(d::DiskDataProvider)  = length(d.ulabels)
stop!(d)                       = (d.reading = false)
Base.wait(d::DiskDataProvider) = wait(d.queue_full)

macro withlock(l, ex)
    quote
        lock($(esc(l)))
        res = $(esc(ex))
        unlock($(esc(l)))
        res
    end
end

withlock(f, l::DiskDataProvider) = withlock(f, l.queuelock)

function withlock(f, l::Base.AbstractLock)
    lock(l)
    try
        return f()
    finally
        unlock(l)
    end
end

function populate_queue(d::DiskDataProvider)
    while d.reading
        y = sample_label(d)
        x = sample_input(d,y)
        xy = (x,y)
        withlock(d) do
            d.queue[d.position] = xy
            d.position += 1
            if d.position > length(d.queue)
                notify(d.queue_full)
                d.position = 1
            end
        end
    end
    @info "Stopped reading"
end


function start_reading(d::DiskDataProvider)
    d.reading = true
    task = @spawn populate_queue(d)
    @info "Populating queue continuosly. Call `stop!(d)` to stop reading`. Call `wait(d)` to be notified when the queue is fully populated."
    task
end


function sample_label(d)
    res = d.label_iterator_state === nothing ?  iterate(d.label_iterator) : iterate(d.label_iterator, d.label_iterator_state)
    res === nothing && error("Reached the end of the label iterator")
    d.label_iterator_state = res[2]
    res[1]
end

function sample_input(d, y)
    files = d.label2files[y]
    fileind = rand(1:length(files))
    x,yr = deserialize(files[fileind])
    x
end


function label2filedict(labels, files)
    ulabels = unique(labels)
    ufiles = map(ulabels) do l
        files[labels .== l]
    end
    label2files = Dict(Pair.(ulabels, ufiles))
end

function sample_random_datapoint(d)
    wait(d.queue_full)
    i = rand(1:length(d.queue))
    d.queue[i]
end

function populate_batch(d,inds)
    for (i,j) in enumerate(inds)
        x,y = d.queue[j]
        d.x_batch[:,:,:,i] .= x
        d.y_batch[i] = y
    end
    (d.x_batch, d.y_batch)
end

function unbuffered_batch(d::DiskDataProvider{XT,YT},inds)::Tuple{Array{eltype(XT),4},Vector{YT}} where {XT,YT}
    mi,ma = extrema(inds)
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(inds))
    Y = similar(d.y_batch, length(inds))
    for (i,j) in enumerate(inds)
        x,y = deserialize(d.files[j])
        X[:,:,:,i] .= x
        Y[i] = y
    end
    X,Y
end

function full_batch(d::DiskDataProvider{XT,YT}) where {XT,YT}
    X = similar(d.x_batch, size(d.x_batch)[1:end-1]..., length(d))
    Y = similar(d.y_batch, length(d))
    for i = 1:length(d)
        x,y = deserialize(d.files[i])
        X[:,:,:,i] .= x
        Y[i] = y
    end
    X,Y
end

struct BufferIterator{T <: DiskDataProvider}
    d::T
end

struct UnbufferedBatchView
    inner
    dataset
end

UnbufferedBatchView(dataset, bs::Int = dataset.batchsize) = UnbufferedBatchView(Iterators.partition(1:length(dataset), bs), BufferIterator(dataset))

function Base.iterate(d::BufferIterator, state=0)
    state == length(d) && return nothing
    (sample_random_datapoint(d),state+1)
end

function Base.iterate(ds::DataSubset{<:DiskDataProvider}, state=nothing)
    inds = ds.indices
    d = ds.data
    populate_batch(d,inds)
end

function Base.iterate(ubw::UnbufferedBatchView)
    res = iterate(ubw.inner)
    res === nothing && return nothing
    unbuffered_batch(ubw.dataset.d, res[1]), res[2]
end

function Base.iterate(ubw::UnbufferedBatchView, state)
    res = iterate(ubw.inner, state)
    res === nothing && return nothing
    unbuffered_batch(ubw.dataset.d, res[1]), res[2]
end

function MLDataUtils.stratifiedobs(d::DiskDataProvider, p::AbstractFloat, args...; kwargs...)
    yt,yv = stratifiedobs(d.labels, p, args...; kwargs...)
    split(d, first(yt.indices), first(yv.indices))
end


Base.length(d::DiskDataProvider) = d.length
Base.length(d::BufferIterator) = d.d.length
Base.length(ubw::UnbufferedBatchView) = length(ubw.inner)

Base.pairs(d::BufferIterator) = enumerate(d)
Base.pairs(d::DiskDataProvider) = enumerate(d)
Base.getindex(d::DiskDataProvider, i) = deserialize(d.files[i])

MLDataUtils.batchview(d::DiskDataProvider) = batchview(BufferIterator(d), size=d.batchsize)
LearnBase.nobs(d::DiskDataProvider)  = length(d.queue)
LearnBase.nobs(d::BufferIterator) = length(d.d.queue)
LearnBase.getobs(d::BufferIterator, inds) = populate_batch(d.d,inds)
LearnBase.datasubset(d::BufferIterator, inds, ::ObsDim.Undefined) = populate_batch(d.d,inds)



end
