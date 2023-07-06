# Changelog

## v0.2.0

- Added support for `min_slope`, `max_slope` in the initial point search for RejectionSampler.
- Added keyword `logdensity` for directly specifying the logdensity in RejectionSampler [#10](https://github.com/mauriciogtec/AdaptiveRejectionSampling.jl/issues/10)
- Improved numerical stability in the exp_integral. Added a warning for instability.
- Added changelog.
- Added complicated example to readme based on [#10](https://github.com/mauriciogtec/AdaptiveRejectionSampling.jl/issues/10)

TODO: Add logger to avoid repeated warnings for numerical instability.