[![Build Status](https://travis-ci.org/baggepinnen/DiskDataProviders.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DiskDataProviders.jl)
[![codecov](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/latest)

# DiskDataProviders.jl

This package implements datastructures that are iterable and backed by a buffer that is fed by data from disk. The reading is done on a separate thread, so make sure Julia is started with at least two threads.

Intended usage: To buffer data reading from disk when training convolutional neural networks (1d or 2d) using [Flux.jl](https://github.com/FluxML/Flux.jl/). This allows the CPU to work with the disk and data while the GPU is working on the training. This package might be useful for other things as well.

For usage example, see [the documentation](https://baggepinnen.github.io/DiskDataProviders.jl/latest)
