var documenterSearchIndex = {"docs":
[{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"(Image: Build Status) (Image: codecov)","category":"page"},{"location":"#DiskDataProviders-1","page":"DiskDataProviders","title":"DiskDataProviders","text":"","category":"section"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"using DiskDataProviders, Test, Serialization, MLDataUtils","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"This package implements datastructures that are iterable and backed by a buffer that is fed by data from disk. If Reading and preproccesing data is faster than one training step, it's recommended to use a ChannelDiskDataProvider, if the training step is fast but reading data takes long time, QueueDiskDataProvider is recommended. Both types do the reading on a separate thread, so make sure Julia is started with at least two threads.","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"My personal use case for this package is training convolutional DL models using Flux. This package does not take care of the transfer of data to the GPU, as I have not managed to do this on a separate thread.","category":"page"},{"location":"#Supervised-vs-unsupervised-1","page":"DiskDataProviders","title":"Supervised vs unsupervised","text":"","category":"section"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"If the task is supervised, you may supply labels using the keyword labels, see example below. If the dataset has labels, it iterates tuples (x,y). If no labels are supplied, it iterates only inputs x. To create an unsupervised dataset with no labels, use Nothing as the label type, e.g. DiskDataProvider{xType, Nothing}.","category":"page"},{"location":"#Usage-example-1","page":"DiskDataProviders","title":"Usage example","text":"","category":"section"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"using DiskDataProviders, Test, Serialization, MLDataUtils\n\n# === Create some random example data ===\ndirpath = mktempdir()*\"/\"\nN = 100\nT = 500\nbatch_size = 2\nqueue_length = 5 # Length of the internal buffer, it's a good idea to make this be some integer multiple of the batch size.\nlabs = rand(1:5, N)\nfor i = 1:N\n    a = randn(T)\n    serialize(dirpath*\"$(i).bin\", (a, labs[i]))\nend\n\nfiles = dirpath .* string.(1:N) .* \".bin\"\n\n# === Create a DiskDataProvider ===\ndataset = ChannelDiskDataProvider{Vector{Float64}, Int}((T,), batch_size, queue_length; labels=labs, files=files)","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"The dataset is iterable and can be used in loops etc. One can also create a BatchView, which is an iterator over batches. The batch size is defined when the DiskDataProvider is created.","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"# === Example usage of the provider ===\ndatasett, datasetv = stratifiedobs(dataset, 0.75)\n\nsort(dataset.ulabels) == 1:5\n\nx,y = first(dataset) # Get one datapoint\n\nt = start_reading(dataset) # this function initiates the reading into the buffer\n\nwait(dataset) # Wait for the reading to start before proceeding\n\nbw = batchview(dataset);\n\nxb,yb = first(bw) # Get the first batch from the buffer\n\nfor (x,y) in bw # Iterate the batches in the batchview\n    # do something with the data\nend\n\nstop!(dataset) # Stop reading into the buffer","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"If your data has more dimensions than 1, e.g., inputs are matrices or 3d-tensors, you create a DiskDataProvider like this","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"dataset = ChannelDiskDataProvider((nrows,ncols,nchannels), batchsize, queuelength; labels=labs, files=files)","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"notice that you have to provide nchannels, which is 1 if the input is a matrix.","category":"page"},{"location":"#Preprocess-data-1","page":"DiskDataProviders","title":"Preprocess data","text":"","category":"section"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"All functionality in this package operates on serialized, preprocessed data files. Serialized files are fast to read, and storing already preprocessed data cuts down on overhead. This package does currently not support arbitrary file formats. The files are read using Julias built in deserializer.","category":"page"},{"location":"#Iterators-1","page":"DiskDataProviders","title":"Iterators","text":"","category":"section"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"If you simply iterate over an AbstractDiskDataProvider, you will iterate over each datapoint in the sequence determined by the vector of file paths. This iteration is not buffered.\nbatchview creates a buffered iterator over batches.\nUnbufferedIterator has the same behaviour as iterating over the AbstractDiskDataProvider (UnbufferedIterator is what is used under the hood).\nBufferedIterator iterates over single datapoints from the buffer.\nfull_batch creates one enormous batch of the entire dataset.","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"Typically, you want to use batchview for training. If you have a small enough dataset (e.g. for validation), you may want to use full_batch, especially if this fits into the GPU memory. Batches are structured according to Flux's notion of a batch, e.g., the last dimension is the batch dimension.","category":"page"},{"location":"#Exported-functions-and-types-1","page":"DiskDataProviders","title":"Exported functions and types","text":"","category":"section"},{"location":"#Index-1","page":"DiskDataProviders","title":"Index","text":"","category":"section"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"","category":"page"},{"location":"#","page":"DiskDataProviders","title":"DiskDataProviders","text":"Modules = [DiskDataProviders, MLDataPattern, MLDataUtils, LearnBase]\nPages = [\"DiskDataProviders.jl\", \"iteration.jl\"]\nPrivate = false","category":"page"},{"location":"#DiskDataProviders.ChannelDiskDataProvider-Union{Tuple{YT}, Tuple{ChannelDiskDataProvider{#s44,YT,BT} where BT where #s44,AbstractArray}} where YT","page":"DiskDataProviders","title":"DiskDataProviders.ChannelDiskDataProvider","text":"ChannelDiskDataProvider(d::ChannelDiskDataProvider, inds::AbstractArray)\n\nThis constructor can be used to create a dataprovider that is a subset of another.\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.ChannelDiskDataProvider-Union{Tuple{YT}, Tuple{XT}, Tuple{Any,Any,Int64}} where YT where XT","page":"DiskDataProviders","title":"DiskDataProviders.ChannelDiskDataProvider","text":"ChannelDiskDataProvider{XT, YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT, YT}\n\nConstructor for ChannelDiskDataProvider. {XT, YT} are the types of the input and output respectively.\n\n#Arguments:\n\nxsize: Tuple with sixe of each data point\nbatchsize: how many datapoints to put in a batch\nqueuelength: length of buffer, it's a good idea to make this be some integer multiple of the batch size.\nkwargs: to set the other fields of the structure.\ntransform : A Function (x,y)->(x,y) or x->x that transforms the data point before it is put in a batch. This can be used to, e.g., apply some pre processing or normalization etc.\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.QueueDiskDataProvider-Union{Tuple{YT}, Tuple{QueueDiskDataProvider{#s44,YT,BT} where BT where #s44,AbstractArray}} where YT","page":"DiskDataProviders","title":"DiskDataProviders.QueueDiskDataProvider","text":"QueueDiskDataProvider(d::QueueDiskDataProvider, inds::AbstractArray)\n\nThis constructor can be used to create a dataprovider that is a subset of another.\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.QueueDiskDataProvider-Union{Tuple{YT}, Tuple{XT}, Tuple{Any,Any,Int64}} where YT where XT","page":"DiskDataProviders","title":"DiskDataProviders.QueueDiskDataProvider","text":"QueueDiskDataProvider{XT, YT}(xsize, batchsize, queuelength::Int; kwargs...) where {XT, YT}\n\nConstructor for QueueDiskDataProvider.\n\n{XT, YT} are the types of the input and output respectively.\n\n#Arguments:\n\nxsize: Tuple with sixe of each data point\nbatchsize: how many datapoints to put in a batch\nqueuelength: length of buffer, it's a good idea to make this be some integer multiple of the batch size.\nkwargs: to set the other fields of the structure.\ntransform : A Function (x,y)->(x,y) or x->x that transforms the data point before it is put in a batch. This can be used to, e.g., apply some pre processing or normalization etc.\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.labels-Tuple{Any}","page":"DiskDataProviders","title":"DiskDataProviders.labels","text":"labels(d)\n\nReturn the labels in the dataset\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.sample_input-Tuple{Any,Any}","page":"DiskDataProviders","title":"DiskDataProviders.sample_input","text":"sample_input(d::AbstractDiskDataProvider, y)\n\nSample one input with label y from the dataset\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.sample_input-Tuple{Any}","page":"DiskDataProviders","title":"DiskDataProviders.sample_input","text":"sample_input(d::AbstractDiskDataProvider)\n\nSample one datapoint from the dataset\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.sample_label-Tuple{Any}","page":"DiskDataProviders","title":"DiskDataProviders.sample_label","text":"sample_label(d)\n\nSample a random label from the dataset\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.start_reading-Tuple{DiskDataProviders.AbstractDiskDataProvider}","page":"DiskDataProviders","title":"DiskDataProviders.start_reading","text":"start_reading(d::AbstractDiskDataProvider)\n\nInitialize reading into the buffer. This function has to be called before the dataset is used. Reading will continue until you call stop! on the dataset. If the dataset is a ChannelDiskDataProvider, this is a non-issue.\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.BufferedIterator","page":"DiskDataProviders","title":"DiskDataProviders.BufferedIterator","text":"struct BufferedIterator{T <: AbstractDiskDataProvider}\n\nCreates an iterator which uses the underlying buffer in the dataset.\n\n\n\n\n\n","category":"type"},{"location":"#DiskDataProviders.UnbufferedIterator","page":"DiskDataProviders","title":"DiskDataProviders.UnbufferedIterator","text":"struct UnbufferedIterator{T <: AbstractDiskDataProvider}\n\nCreates an iterator which does not use the underlying buffer in the dataset.\n\n\n\n\n\n","category":"type"},{"location":"#MLDataPattern.batchview-Tuple{Any}","page":"DiskDataProviders","title":"MLDataPattern.batchview","text":"batchview(d::AbstractDiskDataProvider; size=d.batchsize, kwargs...)\n\nCreate a batch iterator that iterates batches with the batch size defined at the creation of the DiskDataProvider.\n\n\n\n\n\n","category":"method"},{"location":"#DiskDataProviders.full_batch-Tuple{DiskDataProviders.AbstractDiskDataProvider}","page":"DiskDataProviders","title":"DiskDataProviders.full_batch","text":"full_batch(d::AbstractDiskDataProvider)\n\nReturns a matrix with the entire dataset.\n\n\n\n\n\n","category":"method"},{"location":"#MLDataPattern.stratifiedobs-Tuple{DiskDataProviders.AbstractDiskDataProvider,AbstractFloat,Vararg{Any,N} where N}","page":"DiskDataProviders","title":"MLDataPattern.stratifiedobs","text":"stratifiedobs(d::AbstractDiskDataProvider, p::AbstractFloat, args...; kwargs...)\n\nPartition the data into multiple disjoint subsets proportional to the value(s) of p. The observations are assignmed to a data subset using stratified sampling without replacement. These subsets are then returned as a Tuple of subsets, where the first element contains the fraction of observations of data that is specified by the first float in p.\n\nFor example, if p is a Float64 itself, then the return-value will be a tuple with two elements (i.e. subsets), in which the first element contains the fraction of observations specified by p and the second element contains the rest. In the following code the first subset train will contain around 70% of the observations and the second subset test the rest. The key difference to splitobs is that the class distribution in y will actively be preserved in train and test.\n\ntrain, test = stratifiedobs(diskdataprovider, 0.7)\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.nobs-Tuple{Any}","page":"DiskDataProviders","title":"StatsBase.nobs","text":"LearnBase.nobs(d)\n\nGet the number of observations in the dataset\n\n\n\n\n\n","category":"method"}]
}
