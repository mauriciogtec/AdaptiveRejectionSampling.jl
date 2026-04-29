```@meta
CurrentModule = AdaptiveRejectionSampling
```

# AdaptiveRejectionSampling

Documentation for [AdaptiveRejectionSampling](https://github.com/Eliassj/AdaptiveRejectionSampling.jl).

This package implements univariate adaptive rejection sampling as specified by Gilks & Wild [Gilks_Wild_1992](@cite).


## Showcase

AdaptiveRejectionSampling.jl allows you to sample from unnormalized [logarithmically concave](https://en.wikipedia.org/wiki/Logarithmically_concave_function) distributions. In the following example we will sample from an unnormalized [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution).

```@example 1
using AdaptiveRejectionSampling: ARSampler, Objective, sample!
using CairoMakie, Distributions, Random
set_theme!(theme_minimal())
update_theme!(linewidth = 6)
update_theme!(fonts = (; regular = "DejaVu Sans"))
Random.seed!(1)
CairoMakie.activate!(; type="svg") # hide
nothing # hide
```

Lets use a normal distribution centered on $\pi$ with standard deviation 2.

```@example 1
const mu::Float64 = π
const sigma::Float64 = 2.0

# Define a normal distribution without the normalizing terms.
f(x) = exp(-((x - mu)^2) / (2 * sigma^2))

# Use the normal distribution from Distributions.jl for verification.
normal(x) = pdf(Normal(mu, sigma), x)

l, u = mu - sigma * 4, mu + sigma * 4
fig, ax, p = lines(l..u, f, label = "Unnormalized density")
lines!(ax, l..u, normal, label = "Actual normal distribution")
axislegend(ax)
fig # hide
```

Let's imagine we did not know the actual normalizing constant for the normal distribution above. In order to sample from it using AdaptiveRejectionSampling.jl, we first define an objective function using [`AdaptiveRejectionSampling.Objective`](@ref). By default, automatic differentiation through [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is used to calculate the derivative of the function to be sampled from.

First, we need to supply the function in its log form. For illustration purposes, we simply do `log(f(x))`.

```@example 1
f_log(x) = log(f(x))
obj = Objective(f_log)
```

We then define the sampler itself providing the initial points for the hull as well as the desired domain. Initial points needs to be at opposing sides of the function maximum.

```@example 1
sam = ARSampler(obj, (-Inf, Inf), [-2., 4.])
```

In order to retrieve samples, we use the [`AdaptiveRejectionSampling.sample!`](@ref) function. As is indicated by the `!`, this function modifies the sampler itself in order to improve future sampling whenever a sample is rejected.

```@example 1
samples = sample!(sam, 1000);
```

Lets compare the samples drawn from the sampler to the actual target distribution.

```@example 1
fig, ax, p = hist(samples, bins=100, normalization = :pdf, label = "Samples", axis = (; title="1000 samples"))
lines!(l..u, normal, label = "Target distribution", color = :orange, linewidth = 3, alpha = 0.8)
axislegend(ax)
fig # hide
```

Lets do one with more samples in order to verify that it actually approaches the target.

```@example 1
fig, ax, p = hist(sample!(sam, 100000), bins=100, normalization = :pdf, label = "Samples", axis=(; title="100 000 samples")) # hide
lines!(l..u, normal, label = "Target distribution", color = :orange, alpha = 0.8) # hide
axislegend(ax) # hide
fig # hide
```

## Truncated density

```@setup 2
function my_color_theme()
    default_colors = CairoMakie.Makie.wong_colors()
    cycle = Cycle([:color, :strokecolor], covary=true)
    return Theme(
        palette = (;
            strokecolor=default_colors, linestyle=[:solid]),
        Density=(cycle=cycle,)
        )
end
```


```@setup 2
using AdaptiveRejectionSampling: ARSampler, Objective, sample!
using CairoMakie, Distributions, Random # hide
set_theme!(
    merge(
        theme_minimal(),
        my_color_theme()
)) # hide
update_theme!(linewidth = 6) # hide
update_theme!(fonts = (; regular = "DejaVu Sans")) # hide
Random.seed!(1) # hide
CairoMakie.activate!(; type="svg") # hide
```

Lets try a truncated density such as the [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution).

As before, we define the distribution without its normalizing terms as well as a function from `Distributions.jl` to compare with.

```@example 2
const alpha::Float64 = 2.5
const beta::Float64 = 5.0

f(x) = x^(alpha - 1) * (1 - x)^(beta - 1)

beta_proper(x) = pdf(Beta(alpha, beta), x)

l, u = 0, 1
fig, ax, p = lines(l..u, f, label="Unnormalized density")
lines!(ax, l..u, beta_proper, label = "Actual beta distribution")
axislegend(ax)
fig # hide
```

Lets define the objective and sampler. As the beta distribution is bounded on $x \in [0, 1]$ we specify this in the domain argument when creating the sampler.

```@example 2
f_log(x) = log(f(x))
obj = Objective(f_log)

sam = ARSampler(obj, (0.0, 1.0), [0.2, 0.8])
nothing # hide
```

Lets visualize the samples as before.


```@example 2
samples = sample!(sam, 100000)

fig, ax, p = hist(samples, bins=100, label = "Samples", normalization = :pdf)
lines!(ax, 0..1, beta_proper, label = "Target", color = Cycled(2))
axislegend(ax)
fig # hide
```

Of course, we can specify arbitrary bounds. Let's define the same beta distribution as above but truncated at $[0.5, 1.0]$ and compare them with sampling from a truncated `Distributions.jl` distribution (with plotting code hidden for brevity). Since the distribution is bounded to the right of its maximum we only need to specify one initial segment.

```@setup 2
cols = Makie.wong_colors()
elem_ars = LineElement(color = cols[1], linestyle = :solid)
elem_dist = LineElement(color = cols[2], linestyle = :dot)
```

```@example 2
sam_bounded = ARSampler(obj, (0.5, 1.0), [0.8])

samples_bounded = sample!(sam_bounded, 100000)

dist_bounded = truncated(Beta(alpha, beta), lower=0.5)

samples_true = rand(dist_bounded, 100000)

fig, ax, p = density( # hide
    samples_bounded,  # hide
    label = "ARS samples",  # hide
    color = :transparent,  # hide
    strokewidth = 6, # hide
    boundary = (0.0, 1.0) # hide
) # hide
density!( # hide
    ax, # hide
    samples_true,  # hide
    label = "Distributions.jl", # hide
    color = :transparent,  # hide
    strokewidth = 6, # hide
    linestyle = :dot, # hide
    boundary = (0.0, 1.0) # hide
) # hide
Legend( # hide
    fig[1,1], # hide
    [elem_ars, elem_dist], # hide
    ["AdaptiveRejectionSampling.jl", "Distributions.jl"], # hide
    tellheight = false, # hide
    tellwidth = false, # hide
    margin = (10, 10, 10, 10), # hide
    halign = :right, # hide
    valign = :top, # hide
) # hide
fig # hide
```

## BigFloat

Sometimes unnormalized distributions may run the risk of underflowing even with the range provided by `Float64`.
One example of this is found related to an infinite mixture model[rasmussenInfiniteGaussianMixture](@cite) concerning the conditional distribution of the parameter $\alpha$.
The conditional distributions is based on the number of clusters, $k$, and the number of data points, $n$. For low $k$ and $n$ sampling works fine as shown below.

```@setup 3
function my_color_theme()
    default_colors = CairoMakie.Makie.wong_colors()
    cycle = Cycle([:color, :strokecolor], covary=true)
    return Theme(
        palette = (;
            strokecolor=default_colors, linestyle=[:solid]),
        Density=(cycle=cycle,)
        )
end
```


```@example 3
using AdaptiveRejectionSampling: ARSampler, Objective, sample!
using CairoMakie, Distributions, Random, SpecialFunctions, DifferentiationInterface
set_theme!( # hide
        my_color_theme() # hide
) # hide
update_theme!(linewidth = 6) # hide
update_theme!(fonts = (; regular = "DejaVu Sans")) # hide
Random.seed!(1) # hide
CairoMakie.activate!(; type="svg") # hide
```

```@example 3
function log_alpha(x, k, n)
           alpha = exp(x)
           x +
           x * (k - 3 / 2) +
           (-1 / (2 * alpha)) +
           loggamma(alpha) -
           loggamma(n + alpha)
end

obj = Objective(x -> log_alpha(x, 3, 15))

sam = ARSampler(obj, (-Inf, Inf), [-0.2, 0.2])
samples = sample!(sam, 10000)

xticks = -3:3 # hide
fig, ax, p = hist(samples, bins=100, normalization = :pdf, axis = (; title="k = 3, n = 15", xticks = xticks)) # hide
ax2 = Axis(fig[1,1], yaxisposition = :right) # hide
hidespines!(ax2) # hide
hidexdecorations!(ax2) # hide
linkxaxes!(ax, ax2) # hide

lines!(ax2, -3..3, x -> exp(log_alpha(x, 3, 15)), color = :orange, alpha = 0.8) # hide
fig # hide
```

However, for high numbers of clusters/data points we run into issues.

```@example 3
x = log_alpha(0.3, 9, 400)
println("x = ", x)
println("exp(x) = ", exp(x))
```

Using `BigFloat` "solves" the issue:

```@example 3
x = log_alpha(big(0.3), big(9), big(400))
println("x = ", x)
println("exp(x) = ", exp(x))
```

In order to sample using `BigFloat` we pass the type to the sampler constructors.

```@example 3
obj = Objective(x -> log_alpha(x, 9, 400), one(BigFloat))

sam = ARSampler(obj, big.((-Inf, Inf)), big.([-0.2, 0.5]))
samples = sample!(sam, 10000)
nothing # hide
```

...And then plot the result. Note that I had to scale the values of the unnormalized distribution by `m` below in order to get Makie to plot it.
The fit in the plot below (and above) should therefore be taken with a grain of salt but at least the shapes seem to match.

```@example 3
fig, ax, p = hist(samples, bins=100, normalization = :pdf, axis = (; title="k = 9, n = 400", xticks = xticks))
ax2 = Axis(fig[1,1], yaxisposition = :right)
hidespines!(ax2)
hidexdecorations!(ax2)
linkxaxes!(ax, ax2)

m = BigFloat("6.426512129038265565810544883751205877443549713389270459189622783861198714057871e+868")
ex = [exp(log_alpha(big(x), big(9), big(400))) * m for x in -3:0.01:3]

lines!(ax2, -3:0.01:3, ex, color = :orange, alpha = 0.8)
fig # hide
```
