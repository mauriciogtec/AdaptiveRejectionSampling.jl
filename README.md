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

## Sampling from a shifted normal distribution


```julia
# Define function to be sampled
μ, σ = 1.0, 2.0
f(x) = exp(-0.5(x - μ)^2 / σ^2) / sqrt(2pi * σ^2) 
support = (-Inf, Inf)

# Build the sampler and simulate 10,000 samples
sampler = RejectionSampler(f, support, max_segments = 5)
@time sim = run_sampler!(sampler, 10000);
```

      0.010434 seconds (192.15 k allocations: 3.173 MiB)
    

Let's verify the result


```julia
x = -linspace(-10.0, 10.0, 100)
envelop = eval_envelop.(sampler.envelop, x)
target = f.(x)

histogram(sim, normalize = true, label = "Histogram")
plot!(x, [target envelop], width = 2, label = ["Normal(μ, σ)" "Envelop"])
```


![](img/example1.png)


## Let's try a Gamma


```julia
α, β = 5.0, 2.0
f(x) = β^α * x^(α-1) * exp(-β*x) / gamma(α)
support = (0.0, Inf)

sampler = RejectionSampler(f, support)
@time sim = run_sampler!(sampler, 10000) 

x = linspace(0.0, 5.0, 100)
envelop = eval_envelop.(sampler.envelop, x)
target = f.(x)

histogram(sim, normalize = true, label = "Histogram")
plot!(x, [target envelop], width = 2, label = ["Gamma(α, β)" "Envelop"])
```

      0.007299 seconds (182.00 k allocations: 3.027 MiB)
    




![](img/example2.png)

## Truncated distributions and unknown normalisation constant

We don't to provide an exact density--it will sample up to proportionality--and we can do truncated distributions


```julia
α, β = 5.0, 2.0
f(x) = β^α * x^(α-1) * exp(-β*x) / gamma(α)
support = (1.0, 3.5)

sampler = RejectionSampler(f, support)
@time sim = run_sampler!(sampler, 10000) 

x = linspace(0.01, 8.0, 100)
envelop = eval_envelop.(sampler.envelop, x)
target = f.(x)

histogram(sim, normalize = true, label = "histogram")
plot!(x, [target envelop], width = 2, label = ["target density" "envelop"])
```

      0.007766 seconds (181.82 k allocations: 3.024 MiB)
    




![](img/example3.png)