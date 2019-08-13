module DiskDataProviders
using MLDataUtils, DataFrames, Dates

import Threads: nthreads, threadid, @spawn, SpinLock

Base.@kwdef mutable struct DiskDataProvider{XT,YT}
    files
    labels     ::Vector{YT}
    datapoint_size
    label2files::Dict{YT,String}
    queue_full ::Condition = Condition() # QUESTION: should this be a Threads.Condition?
    queue      ::Vector{XT}
    label_iterator
    label_iterator_state = nothing
    position   ::Int = 1
    reading    ::Bool = false
    queuelock  ::SpinLock = SpinLock()
end

function DiskDataProvider(queuelength::Int)

end

nclasses(d::DiskDataProvider) = length(unique(d.labels))

stop(d) = (d.reading = false)

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
        x = sample_input(d,x)
        xy = (x,y)
        withlock(d) do
            queue[d.position] = xy
            d.position += 1
            if d.position > length(d.queue)
                notify(d.queue_full)
                d.position = 1
            end
        end
    end
    @info "Stopped reading"
end

Base.wait(d::DiskDataProvider) = wait(d.queue_full)

function start_reading(d::DiskDataProvider)
    d.reading = true
    task = @spawn populate_queue(d)
    @info "Populating queue continuosly. Call `stop(d)` to stop reading`. Call `wait(d)` to be notified when the queue is fully populated."

    task
end


function sample_label(d)
    res = iterate(d.label_iterator, d.label_iterator_state)
    res === nothing && error("Reached the end of the labl iterator")
    d.label_iterator_state = res[2]
    res[1]
end

function sample_input(d, y)
    files = d.label2files[y]
    fileind = rand(1:length(files))
    load(files[fileind]) # TODO: what if the label appears somewhere in the midle of the file? handle this case by pre-processing?
end


function timerange2label(start,duration)


end


end
