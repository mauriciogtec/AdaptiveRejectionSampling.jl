using AdaptiveRejectionSampling
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(AdaptiveRejectionSampling, :DocTestSetup, :(using AdaptiveRejectionSampling); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))


makedocs(;
    modules = [
    AdaptiveRejectionSampling
    ],
    authors = "Mauricio Tec, Elias Sjölin <elias.sjolin@gmail.com> and contributors",
    sitename = "AdaptiveRejectionSampling.jl",
    format = Documenter.HTML(;
        canonical = "https://mauriciogtec.github.io/AdaptiveRejectionSampling.jl",
        edit_link = "master",
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "Benchmarks" => "benchmarks.md",
        "Public API" => "api.md",
        "Sampler visualization" => "plots.md",
        "Internals" => "devdocs.md",
        "References" => "references.md",
    ],
    plugins = [bib],
    remotes = nothing,
    checkdocs = :public,
    warnonly = [:missing_docs],
)

deploydocs(; repo = "github.com/mauriciogtec/AdaptiveRejectionSampling.jl")
