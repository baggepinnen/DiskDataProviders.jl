module DiskDataProviders
using MLDataUtils, LearnBase, DataFrames, Dates, Serialization

import Base.Threads: nthreads, threadid, @spawn, SpinLock

export DiskDataProvider, label2filedict, start_reading, stop!

Serialization.serialize(filename::AbstractString, data) = open(f->serialize(f, data), filename, "w")
Serialization.deserialize(filename) = open(f->deserialize(f), filename)

Base.@kwdef mutable struct DiskDataProvider{XT,YT}
    batchsize  ::Int           = 8
    length     ::Int           = typemax(Int)
    labels     ::Vector{YT}
    ulabels    ::Vector{YT}    = unique(labels)
    label2files::Dict{YT,Vector{String}}
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

MLDataUtils.nobs(d::DiskDataProvider)  = length(d.queue)
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

function Base.iterate(d::DiskDataProvider, state=0)
    state == length(d) && return nothing
    (sample_random_datapoint(d),state+1)
end
Base.length(d::DiskDataProvider) = d.length

# struct DiskBatches{T,N,XT,YT}
#     d::DiskDataProvider{XT,YT}
#     x::Array{T,N}
#     y::Vector{YT}
# end
#
#
# function Base.iterate(db::DiskBatches, state=0)
#     for (i,(x,y)) = enumerate(Iterators.take(db.d, size(db.x, 4)))
#         db.x[:,:,1,i] .= x
#         db.y[i] = y
#     end
#     ((db.x, db.y), state+1)
# end
# Base.length(d::DiskBatches) = typemax(Int)


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

function Base.iterate(ds::DataSubset{<:DiskDataProvider}, state=nothing)
    inds = ds.indices
    d = ds.data
    populate_batch(d,inds)
end

MLDataUtils.batchview(d::DiskDataProvider) = batchview(d, size=d.batchsize)

LearnBase.getobs(d::DiskDataProvider, inds) = populate_batch(d,inds)
LearnBase.nobs(d::DiskDataProvider) = length(d.queue)
LearnBase.datasubset(d::DiskDataProvider, inds, ::ObsDim.Undefined) = populate_batch(d,inds)

end
