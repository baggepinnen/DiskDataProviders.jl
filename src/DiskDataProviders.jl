module DiskDataProviders
using MLDataUtils, DataFrames, Dates, Serialization

import Base.Threads: nthreads, threadid, @spawn, SpinLock

export DiskDataProvider2, label2filedict, start_reading, stop

Serialization.serialize(filename::AbstractString, data) = open(f->serialize(f, data), filename, "w")
Serialization.deserialize(filename) = open(f->deserialize(f), filename)

Base.@kwdef mutable struct DiskDataProvider2{XT,YT}
    length     ::Int = typemax(Int)
    labels     ::Vector{YT}
    label2files::Dict{YT,Vector{String}}
    queue_full ::Threads.Event = Threads.Event()
    queue      ::Vector{Tuple{XT,YT}}
    label_iterator
    label_iterator_state = nothing
    position   ::Int = 1
    reading    ::Bool = false
    queuelock  ::SpinLock = SpinLock()
end

function DiskDataProvider2{XT,YT}(queuelength::Int; kwargs...) where {XT,YT}
    DiskDataProvider2{XT,YT}(; queue=Vector{Tuple{XT,YT}}(undef,queuelength), kwargs...)
end

nclasses(d::DiskDataProvider2) = length(unique(d.labels))

stop(d) = (d.reading = false)

macro withlock(l, ex)
    quote
        lock($(esc(l)))
        res = $(esc(ex))
        unlock($(esc(l)))
        res
    end
end

withlock(f, l::DiskDataProvider2) = withlock(f, l.queuelock)

function withlock(f, l::Base.AbstractLock)
    lock(l)
    try
        return f()
    finally
        unlock(l)
    end
end


function populate_queue(d::DiskDataProvider2)
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

Base.wait(d::DiskDataProvider2) = wait(d.queue_full)

function start_reading(d::DiskDataProvider2)
    d.reading = true
    task = @spawn populate_queue(d)
    @info "Populating queue continuosly. Call `stop(d)` to stop reading`. Call `wait(d)` to be notified when the queue is fully populated."

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

function Base.iterate(d::DiskDataProvider2, state=0)
    state == length(d) && return nothing
     (sample_random_datapoint(d),state+1)
end
Base.length(d::DiskDataProvider2) = d.length

function sample_random_datapoint(d)
    wait(d.queue_full)
    i = rand(1:length(d.queue))
    d.queue[i]
end

end
