
# Benchmarks

```@example bench1
using AdaptiveRejectionSampling: ARSampler, Objective, sample!
using DifferentiationInterface, Distributions
using ForwardDiff
using Mooncake
using Enzyme
using Chairmarks

# Define function to sample from
f(x) = logpdf(Laplace(0., 0.5), x) + logpdf(Normal(0.0, 2.0), x)
nothing # hide
```

## ForwardDiff.jl

```@example bench1
const sam_forwarddiff = ARSampler(Objective(f), (-Inf, Inf), [-0.5, 0.5])

@be deepcopy(sam_forwarddiff) sample!(_, 100000, max_segments = 25) samples=100 evals=1 seconds=1000
```

## Mooncake.jl

```@example bench1
sam_mooncake = ARSampler(Objective(f; adbackend = AutoMooncake()), (-Inf, Inf), [-0.5, 0.5])

@be deepcopy(sam_mooncake) sample!(_, 100000, max_segments = 25) samples=100 evals=1 seconds=1000
```

## Enzyme

```@example bench1
sam_mooncake = ARSampler(Objective(f; adbackend = AutoEnzyme()), (-Inf, Inf), [-0.5, 0.5])

@be deepcopy(sam_mooncake) sample!(_, 100000, max_segments = 25) samples=100 evals=1 seconds=1000
```
