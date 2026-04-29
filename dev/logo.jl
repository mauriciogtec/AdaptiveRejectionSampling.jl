using CairoMakie, Distributions, Colors


function mklogo(fs)
    vs = [x -> pdf(Gamma(exp(s), s), x) for s in 0.9:0.1:2]
    cols = repeat([Colors.JULIA_LOGO_COLORS...], 3)
    f = Figure(size=(500, 500), figure_padding=0, backgroundcolor=:transparent)
    ax = Axis(f[1, 1], backgroundcolor=:transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    for i in eachindex(fs)
        lines!(0 .. 8, fs[i], linewidth=9, color=cols[i])
    end
    f
end

mklogo(vs)
