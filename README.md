[![Build Status](https://travis-ci.org/mauriciogtec/AdaptiveRejectionSampling.jl.svg?branch=master)](https://travis-ci.org/mauriciogtec/AdaptiveRejectionSampling.jl)

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