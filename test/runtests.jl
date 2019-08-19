using DiskDataProviders, Test
dataset = ChannelDiskDataProvider{Matrix{Float32}, String}((10001,300,1), 8, 104; labels=labels, files=files)
datasett, datasetv = stratifiedobs(dataset, 0.75)

@test intersect(datasett.files, datasetv.files) == []
@test Set(union(datasett.files, datasetv.files)) == Set(dataset.files)

# @btime dataset[rand(1:length(dataset))]
