module DiskDataProviders
using MLDataUtils, LearnBase, Dates, Serialization

import Base.Threads: nthreads, threadid, @spawn, SpinLock

export QueueDiskDataProvider, ChannelDiskDataProvider, label2filedict, start_reading, stop!, BufferedIterator, UnbufferedIterator, labels

# Serialization.serialize(filename::AbstractString, data) = open(f->serialize(f, data), filename, "w")
# Serialization.deserialize(filename) = open(f->deserialize(f), filename)

abstract type AbstractDiskDataProvider{XT,YT} end


Base.@kwdef mutable struct QueueDiskDataProvider{XT,YT,BT} <: AbstractDiskDataProvider{XT,YT}
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
    transform = nothing
end

"""
    QueueDiskDataProvider{XT, YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT, YT}

Constructor for QueueDiskDataProvider.

#Arguments:
- `xsize`: Tuple with sixe of each data point
- `batchsize`: how many datapoints to put in a batch
- `queuelength`: length of buffer
- `kwargs`: to set the other fields of the structure.
"""
function QueueDiskDataProvider{XT,YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT,YT}
    queue   = Vector{Tuple{XT,YT}}(undef, queuelength)
    if XT <: AbstractVector
        x_batch = Array{Float32,2}(undef, xsize..., batchsize)
    else
        x_batch = Array{Float32,4}(undef, xsize..., batchsize)
    end
    y_batch = Vector{YT}(undef, batchsize)
    QueueDiskDataProvider{XT,YT,typeof(x_batch)}(;
        queue     = queue,
        batchsize = batchsize,
        x_batch = x_batch,
        y_batch = y_batch,
        kwargs...)
end

"""
    QueueDiskDataProvider(d::QueueDiskDataProvider, inds::AbstractArray)

DOCSTRING
"""
function QueueDiskDataProvider(d::QueueDiskDataProvider, inds::AbstractArray)
    QueueDiskDataProvider(
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
        y_batch              = similar(d.y_batch),
        transform            = d.transform
    )
end

Base.@kwdef mutable struct ChannelDiskDataProvider{XT,YT,BT} <: AbstractDiskDataProvider{XT,YT}
    batchsize  ::Int           = 8
    labels     ::Vector{YT}
    files      ::Vector{String}
    length     ::Int           = length(labels)
    ulabels    ::Vector{YT}    = unique(labels)
    label2files::Dict{YT,Vector{String}} = label2filedict(labels, files)
    channel    ::Channel{Tuple{XT,YT}}
    label_iterator             = Iterators.cycle(unique(labels))
    label_iterator_state       = nothing
    reading    ::Bool          = false
    queuelock  ::SpinLock      = SpinLock()
    x_batch    ::BT
    y_batch    ::Vector{YT}
    transform = nothing
end

"""
    ChannelDiskDataProvider{XT, YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT, YT}

Constructor for ChannelDiskDataProvider.

#Arguments:
- `xsize`: Tuple with sixe of each data point
- `batchsize`: how many datapoints to put in a batch
- `queuelength`: length of buffer
- `kwargs`: to set the other fields of the structure.
"""
function ChannelDiskDataProvider{XT,YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT,YT}
    channel = Channel{Tuple{XT,YT}}(queuelength)
    if XT <: AbstractVector
        x_batch = Array{Float32,2}(undef, xsize..., batchsize)
    else
        x_batch = Array{Float32,4}(undef, xsize..., batchsize)
    end
    y_batch = Vector{YT}(undef, batchsize)
    ChannelDiskDataProvider{XT,YT,typeof(x_batch)}(;
        channel   = channel,
        batchsize = batchsize,
        x_batch = x_batch,
        y_batch = y_batch,
        kwargs...)
end

"""
    ChannelDiskDataProvider(d::ChannelDiskDataProvider, inds::AbstractArray)

DOCSTRING
"""
function ChannelDiskDataProvider(d::ChannelDiskDataProvider, inds::AbstractArray)
    ChannelDiskDataProvider(
        batchsize            = d.batchsize,
        length               = length(inds),
        labels               = d.labels[inds],
        files                = d.files[inds],
        ulabels              = d.ulabels,
        channel              = deepcopy(d.channel),
        label_iterator       = d.label_iterator,
        label_iterator_state = nothing,
        reading              = false,
        queuelock            = SpinLock(),
        x_batch              = similar(d.x_batch),
        y_batch              = similar(d.y_batch),
        transform            = d.transform
    )
end

for T in (:QueueDiskDataProvider, :ChannelDiskDataProvider)
    @eval begin
        function Base.split(d::$(T),i1,i2)
            $(T)(d,i1), $(T)(d,i2)
        end
    end
end

Base.show(io::IO, d::AbstractDiskDataProvider) = println(io, "$(typeof(d)), length: $(length(d))")

"""
    queuelength(d)

How long queue (buffer) does the dataset hold
"""
queuelength(d)


"""
    labels(d)

Return the labels in the dataset
"""
labels(d)

queuelength(d::QueueDiskDataProvider) = length(d.queue)
queuelength(d::ChannelDiskDataProvider) = d.channel.sz_max
labels(d     ::AbstractDiskDataProvider) = map(findfirst, eachrow(d.labels .== reshape(d.ulabels,1,:)))
labels(d     ::Vector{<:Tuple})  = last.(d)
nclasses(d   ::AbstractDiskDataProvider) = length(d.ulabels)
stop!(d)                         = (d.reading = false)
Base.wait(d  ::QueueDiskDataProvider) = wait(d.queue_full)
Base.wait(d  ::ChannelDiskDataProvider) = wait(d.channel)
Base.take!(d ::ChannelDiskDataProvider) = take!(d.channel)

Base.isready(d  ::QueueDiskDataProvider) = isready(d.queue_full.set)
Base.isready(d  ::ChannelDiskDataProvider) = isready(d.channel)

macro withlock(l, ex)
    quote
        lock($(esc(l)))
        res = $(esc(ex))
        unlock($(esc(l)))
        res
    end
end

withlock(f, l::AbstractDiskDataProvider) = withlock(f, l.queuelock)

function withlock(f, l::Base.AbstractLock)
    lock(l)
    try
        return f()
    finally
        unlock(l)
    end
end

function get_xy(d::AbstractDiskDataProvider)
    y = sample_label(d)
    x = sample_input(d,y)
    xy = (x,y)
end

function get_xy(d::AbstractDiskDataProvider{XT,Nothing}) where {XT}
    x = sample_input(d)
    (x,nothing)
end

function populate(d::QueueDiskDataProvider)
    while d.reading
        xy = get_xy(d)
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

function populate(d::ChannelDiskDataProvider)
    while d.reading
        xy = get_xy(d)
        put!(d.channel, xy)
    end
    @info "Stopped reading"
end

"""
    start_reading(d::AbstractDiskDataProvider)

Initialize reading into the buffer. This function has to be called before the dataset is used. Reading will continue until you call `stop!` on the dataset. If the dataset is a [`ChannelDiskDataProvider`](@ref), this is a non-issue.
"""
function start_reading(d::AbstractDiskDataProvider)
    d.reading = true
    task = @spawn populate(d)
    @info "Populating queue continuosly. Call `stop!(d)` to stop reading`. Call `wait(d)` to be notified when the queue is fully populated."
    task
end

"""
    sample_label(d)

Sample a random label from the dataset
"""
function sample_label(d)
    res = d.label_iterator_state === nothing ?  iterate(d.label_iterator) : iterate(d.label_iterator, d.label_iterator_state)
    res === nothing && error("Reached the end of the label iterator")
    d.label_iterator_state = res[2]
    res[1]
end

"""
    sample_input(d::AbstractDiskDataProvider, y)

Sample one input with label `y` from the dataset
"""
function sample_input(d, y)
    files = d.label2files[y]
    fileind = rand(1:length(files))
    x,yr = read_and_transform(d,fileind)
    x
end

"""
    sample_input(d::AbstractDiskDataProvider)

Sample one input from the dataset
"""
function sample_input(d)
    fileind = rand(1:length(d.files))
    x = read_and_transform(d,fileind)
    x
end


function label2filedict(labels, files)
    ulabels = unique(labels)
    ufiles = map(ulabels) do l
        files[labels .== l]
    end
    label2files = Dict(Pair.(ulabels, ufiles))
end

"""
    sample_random_datapoint(d)

DOCSTRING
"""
function sample_random_datapoint(d)
    wait(d.queue_full)
    i = rand(1:length(d.queue))
    d.queue[i]
end


include("iteration.jl")


end
