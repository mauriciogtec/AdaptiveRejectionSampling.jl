import AdaptiveRejectionSampling
using ForwardDiff
m = AdaptiveRejectionSampling

f(x) = 1 / √(2π) * exp(-0.5x^2)
# g(x) = ForwardDiff.derivative(f, x)
# h(x) = ForwardDiff.derivative(g, x)

envelop = m.initialize_envelop(f, -1.0, 1.0)

m.initialize_envelop(f, 2.0, 1.0)
 m.initialize_envelop(f, 0.5, 1.0)
