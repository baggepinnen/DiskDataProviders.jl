module DiskDataProviders
using Dates, Serialization, Random, ResumableFunctions

import Base.Threads: nthreads, threadid, @spawn, SpinLock
using Base.Iterators: cycle, peel, partition, take

export QueueDiskDataProvider, ChannelDiskDataProvider, label2filedict, start_reading, stop!, buffered, unbuffered, batchview, unbuffered_batchview, labels, sample_input, sample_label, full_batch, shuffle, shuffle!

import MLDataUtils
using MLDataUtils: stratifiedobs
export stratifiedobs, batchview

# Serialization.serialize(filename::AbstractString, data) = open(f->serialize(f, data), filename, "w")
# Serialization.deserialize(filename) = open(f->deserialize(f), filename)

abstract type AbstractDiskDataProvider{XT,YT} end


Base.@kwdef mutable struct QueueDiskDataProvider{XT,YT,BT} <: AbstractDiskDataProvider{XT,YT}
    batchsize  ::Int           = 8
    files      ::Vector{String}
    labels     ::Vector{YT}    = [nothing]
    length     ::Int           = length(files) # Beware, this prevents the use of function length below this entry
    ulabels    ::Vector{YT}    = unique(labels)
    label2files::Dict{YT,Vector{String}} = label2filedict(labels, files)
    queue_full ::Threads.Event = Threads.Event()
    queue      ::Vector{Tuple{XT,YT}}
    label_iterator             = cycle(unique(labels))
    label_iterator_state       = nothing
    file_iterator              = cycle(randperm(length))
    position   ::Int           = 1
    reading    ::Bool          = false
    queuelock  ::SpinLock      = SpinLock()
    x_batch    ::BT
    y_batch    ::Vector{YT}
    transform                  = nothing
end


"""
    QueueDiskDataProvider{XT, YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT, YT}

Constructor for QueueDiskDataProvider.

`{XT, YT}` are the types of the input and output respectively.

#Arguments:
- `xsize`: Tuple with size of each data point
- `batchsize`: how many datapoints to put in a batch
- `queuelength`: length of buffer, it's a good idea to make this be some integer multiple of the batch size.
- `kwargs`: to set the other fields of the structure.
- `transform` : A Function `(x,y)->(x,y)` or `x->x` that transforms the data point before it is put in a batch. This can be used to, e.g., apply some pre processing or normalization etc.
"""
function QueueDiskDataProvider{XT,YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT,YT}
    queue   = Vector{Tuple{XT,YT}}(undef, queuelength)
    if XT <: AbstractVector
        x_batch = Array{Float32,2}(undef, xsize..., batchsize)
    else
        x_batch = Array{Float32,4}(undef, xsize..., batchsize)
    end
    y_batch = Vector{YT}(undef,  YT === Nothing ? 0 : batchsize)
    QueueDiskDataProvider{XT,YT,typeof(x_batch)}(;
        queue     = queue,
        batchsize = batchsize,
        x_batch = x_batch,
        y_batch = y_batch,
        kwargs...)
end


"""
    QueueDiskDataProvider(d::QueueDiskDataProvider, inds::AbstractArray)

This constructor can be used to create a dataprovider that is a subset of another.
"""
function QueueDiskDataProvider(d::QueueDiskDataProvider{<:Any, YT}, inds::AbstractArray) where YT
    QueueDiskDataProvider(
        batchsize            = d.batchsize,
        labels               = YT === Nothing ? [nothing] : d.labels[inds],
        files                = d.files[inds],
        length               = length(inds),
        ulabels              = d.ulabels,
        queue_full           = Threads.Event(),
        queue                = similar(d.queue),
        label_iterator       = d.label_iterator,
        label_iterator_state = nothing,
        file_iterator        = cycle(randperm(length(inds))),
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
    labels     ::Vector{YT}    = [nothing]
    files      ::Vector{String}
    length     ::Int           = length(files) # Beware, this prevents the use of function length below this entry
    ulabels    ::Vector{YT}    = unique(labels)
    label2files::Dict{YT,Vector{String}} = label2filedict(labels, files)
    channel    ::Channel{Tuple{XT,YT}}
    label_iterator             = cycle(unique(labels))
    label_iterator_state       = nothing
    file_iterator              = cycle(randperm(length))
    reading    ::Bool          = false
    queuelock  ::SpinLock      = SpinLock()
    x_batch    ::BT
    y_batch    ::Vector{YT}
    transform = nothing
end

"""
    ChannelDiskDataProvider{XT, YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT, YT}

Constructor for ChannelDiskDataProvider.
`{XT, YT}` are the types of the input and output respectively.

#Arguments:
- `xsize`: Tuple with size of each data point
- `batchsize`: how many datapoints to put in a batch
- `queuelength`: length of buffer, it's a good idea to make this be some integer multiple of the batch size.
- `kwargs`: to set the other fields of the structure.
- `transform` : A Function `(x,y)->(x,y)` or `x->x` that transforms the data point before it is put in a batch. This can be used to, e.g., apply some pre processing or normalization etc.
"""
function ChannelDiskDataProvider{XT,YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT,YT}
    channel = Channel{Tuple{XT,YT}}(queuelength)
    if XT <: AbstractVector
        x_batch = Array{Float32,2}(undef, xsize..., batchsize)
    else
        x_batch = Array{Float32,4}(undef, xsize..., batchsize)
    end
    y_batch = Vector{YT}(undef, YT === Nothing ? 0 : batchsize)
    ChannelDiskDataProvider{XT,YT,typeof(x_batch)}(;
        channel   = channel,
        batchsize = batchsize,
        x_batch = x_batch,
        y_batch = y_batch,
        kwargs...)
end

"""
    ChannelDiskDataProvider(d::ChannelDiskDataProvider, inds::AbstractArray)

This constructor can be used to create a dataprovider that is a subset of another.
"""
function ChannelDiskDataProvider(d::ChannelDiskDataProvider{<:Any, YT}, inds::AbstractArray) where YT
    ChannelDiskDataProvider(
        batchsize            = d.batchsize,
        length               = length(inds),
        labels               = YT === Nothing ? [nothing] : d.labels[inds],
        files                = d.files[inds],
        ulabels              = d.ulabels,
        channel              = deepcopy(d.channel),
        label_iterator       = d.label_iterator,
        label_iterator_state = nothing,
        file_iterator        = cycle(randperm(length(inds))),
        reading              = false,
        queuelock            = SpinLock(),
        x_batch              = similar(d.x_batch),
        y_batch              = similar(d.y_batch),
        transform            = d.transform
    )
end

for T in (:QueueDiskDataProvider, :ChannelDiskDataProvider)
    @eval begin
        """
            Base.split(d::AbstractDiskDataProvider,i1,i2)

        Split the dataset into two parts defined by vectors of indices
        """
        function Base.split(d::$(T),i1,i2)
            $(T)(d,i1), $(T)(d,i2)
        end
        """
            Base.getindex(d::AbstractDiskDataProvider, inds::AbstractArray)
        Get a dataset corresponding to a subset of the file indices
        """
        Base.getindex(d::$(T), inds::AbstractArray) = $(T)(d, inds)
    end
end

"""
    Random.shuffle!(d::AbstractDiskDataProvider)

Shuffle the file order in place.
"""
Random.shuffle!(d::AbstractDiskDataProvider) = shuffle!(d.files)

"""
    Random.shuffle(d::AbstractDiskDataProvider)

Return a new dataset with the file order shuffled
"""
function Random.shuffle(d::AbstractDiskDataProvider)
    d = deepcopy(d)
    shuffle!(d.files)
    d
end


Base.show(io::IO, d::AbstractDiskDataProvider) = println(io, "$(typeof(d)), length: $(length(d))")

"""
    queuelength(d)

How long queue (buffer) does the dataset hold? Call this function and you'll know
"""
queuelength(d)


"""
    labels(d)

Return numeric labes in the dataset, i.e., strings are converted to integers etc.
"""
labels(d)

queuelength(d::QueueDiskDataProvider) = length(d.queue)
queuelength(d::ChannelDiskDataProvider) = d.channel.sz_max
labels(d     ::AbstractDiskDataProvider) = map(findfirst, eachrow(d.labels .== reshape(d.ulabels,1,:)))
labels(d     ::Vector{<:Tuple})  = last.(d)
nclasses(d   ::AbstractDiskDataProvider) = length(d.ulabels)
stop!(d)                         = (d.reading = false)


"""
    Base.wait(d::AbstractDiskDataProvider)

After having called [`start_reading`](@ref), you may call `wait` on the dataset. This will block until the buffer is ready to be read from.
"""
Base.wait(d::AbstractDiskDataProvider)

"""
    Base.isready(d::AbstractDiskDataProvider)

After having called [`start_reading`](@ref), you may call `isready` on the dataset. This will tell you if the buffer is ready to be read from. See also [`wait`](@ref)
"""
Base.isready(d::AbstractDiskDataProvider)

Base.wait(d  ::QueueDiskDataProvider) = wait(d.queue_full)
Base.wait(d  ::ChannelDiskDataProvider) = wait(d.channel)
Base.take!(d ::ChannelDiskDataProvider) = take!(d.channel)

Base.isready(d  ::QueueDiskDataProvider) = d.queue_full.set
Base.isready(d  ::ChannelDiskDataProvider) = isready(d.channel)


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
        yield() # This is required for tests to pass on travis
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
    xy = read_and_transform(d,fileind)
    maybefirst(xy)
end

"""
    sample_input(d::AbstractDiskDataProvider)

Sample one datapoint from the dataset
"""
function sample_input(d)
    fileind, d.file_iterator = peel(d.file_iterator)
    xy = read_and_transform(d,fileind)
    maybefirst(xy)
end

@inline maybefirst(x) = x
@inline maybefirst(x::Tuple) = first(x)

function label2filedict(labels, files)
    eltype(labels) == Nothing && return Dict(nothing=>files)
    ulabels = unique(labels)
    ufiles = map(ulabels) do l
        files[labels .== l]
    end
    label2files = Dict(Pair.(ulabels, ufiles))
end

function sample_random_datapoint(d::QueueDiskDataProvider)
    wait(d.queue_full)
    i = rand(1:length(d.queue))
    d.queue[i]
end


include("iteration.jl")


end
