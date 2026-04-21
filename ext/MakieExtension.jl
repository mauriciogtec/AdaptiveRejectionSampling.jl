module MakieExtension
using Makie
using AdaptiveRejectionSampling
import AdaptiveRejectionSampling: hullplot, hullplot!, eval_hull, abscissae, ARSampler

@recipe Hullplot begin
    abscissae = true
    target = false
    upper_hull = true
    lower_hull = true
    target_linewidth = @inherit linewidth
    upper_hull_linewidth = @inherit linewidth
    lower_hull_linewidth = @inherit linewidth
    target_label = "Target"
    upper_hull_label = "Upper hull"
    lower_hull_label = "Lower hull"
    abscissae_label = "Abscissae"
end


function Makie.plot!(p::Hullplot{<:Tuple{<:Any, <:ARSampler}})

    sam = p[2][]
    upper_hull = sam.upper_hull
    lower_hull = sam.lower_hull
    obj = sam.objective
    xs = p[1]

    i = 1
    if p.lower_hull[]
        lines!(
            p,
            xs,
            x -> exp(eval_hull(lower_hull, x)),
            color=Cycled(i),
            linewidth = p.lower_hull_linewidth,
            label = p.lower_hull_label
        )
        i += 1
    end
    if p.upper_hull[]
        lines!(
            p,
            xs,
            x -> exp(eval_hull(upper_hull, x)),
            linewidth = p.upper_hull_linewidth,
            color=Cycled(i),
            label = p.upper_hull_label
        )
        i += 1
    end
    if p.target[]
        lines!(
            p,
            xs,
            x -> exp(obj.f(x)),
            color=Cycled(i),
            label = p.target_label,
            linewidth = p.target_linewidth
        )
        i += 1
    end
    if p.abscissae[]
        absc = abscissae(upper_hull)
        scatter!(
            p,
            absc,
            exp.(eval_hull.(upper_hull, absc)),
            color = Cycled(i),
            label = p.abscissae_label
        )
    end

end

function Makie.get_plots(plot::Hullplot)
    return plot.plots
end

end # End of module
