using JLD
using Plots

data = load("transcriptomics_data.jld")
coordinates = data["coordinates"]
M = data["count_matrix"]
true_labels = data["true_labels"]
label_names = data["label_names"]

m,n = 4180,4021 # number of cells x number of genes (features)

# Each row of M is the values for a gene
# Ex. M[1,:] gives the values for the first gene
# The different columns corrispond to the coordinates the genes are sampled at:
ys, xs = coordinates[1,:], -coordinates[2,:]

xlims = [extrema(xs)...]
ylims = [extrema(ys)...]

begin
gene = 1
# Plot the coordinates with darker points indicating larger values,
scatter(xs,ys;
    markeralpha=M[gene,:] ./ maximum(M[gene,:])
    ) |> display
end

# All on one plot
p= scatter(
    legend=:outerright, axis=false, grid=false, aspect_ratio=:equal,size=(600,300),widen=false, xlims=(-5,160))

for (label_index, label_name, markercolor) in zip(0:11, label_names, palette(:Paired_12))
    xs_restricted = xs[true_labels .== label_index]
    ys_restricted = ys[true_labels .== label_index]
    scatter!(xs_restricted, ys_restricted;
        label=label_name,
        markercolor,
        markersize=4,
        markerstrokewidth=0,
        markeralpha=1
        )
end
display(p)
# savefig(p, "../figures/transcriptomics/all_true_labels.svg")

# Separate plots
options = (:legend=>false, :ticks=>false, :grid=>false,
# :aspect_ratio=>:equal,
framestyle = :box, :size => (480,320))

for (label_index, label_name, markercolor) in zip(0:11, label_names, palette(:Paired_12))
    xs_restricted = xs[true_labels .== label_index]
    ys_restricted = ys[true_labels .== label_index]
    p = scatter(xs_restricted, ys_restricted;
        title=label_name,
        titlefontsize = 18,
        markercolor,
        markersize=6,
        markerstrokewidth=0,
        markeralpha=1,
        xlims,
        ylims,
        options...
        )
    display(p)
    # savefig(p, "../figures/transcriptomics/true_cell_$label_name.svg")
end
