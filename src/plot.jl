# Function definitions for use in Makie extension

"""
    hullplot(x, sampler, ...)

Plots the sampler (upper/lower hulls and abscissae) as well as the target function.

# Keyword arguments

- `abscissae` = true
- `target` = false
- `upper_hull` = true
- `lower_hull` = true
- `target_linewidth` = @inherit linewidth
- `upper_hull_linewidth` = @inherit linewidth
- `lower_hull_linewidth` = @inherit linewidth
- `target_label` = "Target"
- `upper_hull_label` = "Upper hull"
- `lower_hull_label` = "Lower hull"
- `abscissae_label` = "Abscissae"

"""
function hullplot end

"""
    hullplot!([ax, ], x, sampler, ...)

See [`AdaptiveRejectionSampling.hullplot`](@ref)
"""
function hullplot! end
