[![Build Status](https://travis-ci.org/baggepinnen/DiskDataProviders.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DiskDataProviders.jl)
[![codecov](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/DiskDataProviders.jl)


<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/stable) -->
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/DiskDataProviders.jl/latest)

# DiskDataProviders.jl

This package implements datastructures that are iterable and backed by a buffer that is fed by data from disc. The reading is done on a separate thread, so make sure Julia is started with at least two threads.

For usage example, see [the documentation](https://baggepinnen.github.io/DiskDataProviders.jl/latest)
