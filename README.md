[![Build Status](https://travis-ci.org/mauriciogtec/AdaptiveRejectionSampling.jl.svg?branch=master)](https://travis-ci.org/mauriciogtec/AdaptiveRejectionSampling.jl)
[![AdaptiveRejectionSampling](http://pkg.julialang.org/badges/AdaptiveRejectionSampling_0.6.svg)](http://pkg.julialang.org/detail/AdaptiveRejectionSampling)
[![AdaptiveRejectionSampling](http://pkg.julialang.org/badges/AdaptiveRejectionSampling_0.7.svg)](http://pkg.julialang.org/detail/AdaptiveRejectionSampling)

[![Coverage Status](https://coveralls.io/repos/github/mauriciogtec/AdaptiveRejectionSampling.jl/badge.svg?branch=master)](https://coveralls.io/github/mauriciogtec/AdaptiveRejectionSampling.jl?branch=master)
[![codecov](https://codecov.io/gh/mauriciogtec/AdaptiveRejectionSampling.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mauriciogtec/AdaptiveRejectionSampling.jl)

[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.png?v=103)](https://opensource.org/licenses/mit-license.php)

# AdaptiveRejectionSampling

```julia
using AdaptiveRejectionSampling
using Plots
```


```julia
μ, σ = 1.0, 2.0
f(x) = exp(-0.5(x - μ)^2 / σ^2) / sqrt(2pi * σ^2);
```


```julia
sampler = RejectionSampler(f); # domain and max segment should be here
```


```julia
@time sim = run_sampler!(sampler, 10000; max_segments = 5) #10,000 samples;
```

      0.155764 seconds (773.26 k allocations: 28.453 MiB, 3.92% gc time)



```julia
env = eval_envelop.(sampler.envelop, x)
p = histogram(sim, normalize = true, label = "histogram")
plot!(p, x, [f.(x) env], width = 2, label = ["target density" "envelop"])
```

![](./img/example1.png)
