using DiskDataProviders, Test, Serialization, MLDataUtils



@time @testset "DiskDataProviders" begin
    dirpath = mktempdir()*"/"
    N = 100
    T = 500
    bs = 2
    labs = rand(1:5, N)
    for i = 1:N
        a = randn(T)
        serialize(dirpath*"$(i).bin", (a, labs[i]))
    end

    files = dirpath .* string.(1:N) .* ".bin"
    TYPE = QueueDiskDataProvider{Vector{Float64}, Int}
    @info "Testing DiskDataProviders"


    for TYPE in [ChannelDiskDataProvider{Vector{Float64}, Int}, QueueDiskDataProvider{Vector{Float64}, Int}]

        println("Testing $TYPE")
        dataset = TYPE((T,), bs, 5; labels=labs, files=files)
        datasett, datasetv = stratifiedobs(dataset, 0.75)

        @test intersect(datasett.files, datasetv.files) == []
        @test Set(union(datasett.files, datasetv.files)) == Set(dataset.files)

        @test sort(dataset.ulabels) == 1:5

        cdata = collect(dataset)
        @test length(cdata) == N

        @test_throws ErrorException batchview(dataset)
        @test length.(first(dataset)) == (T,1)

        t = start_reading(dataset)
        wait(dataset)

        bw = batchview(dataset)
        @test length(bw) == N ÷ bs
        @test size.(first(bw)) == ((T,bs), (bs,))

        stop!(dataset)
        println("Done with $TYPE")
    end

end

println("Done")
