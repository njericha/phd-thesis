using KernelDensity
using Plots
using LinearAlgebra: norm
using BlockTensorFactorization
using Random
using JLD

using Statistics: mean # used by BlockTensorFactorization

# Define two kernel density estimation helpers
"""
    repeatcoord(coordinates, values)

Repeats coordinates the number of times given by values.
Both lists should be the same length.

Example
-------
coordinates = [(0,0), (1,1), (1,2)]
values = [1, 3, 2]
repeatcoord(coordinates, values)

[(0,0), (1,1), (1,1), (1,1), (1,2), (1,2)]
"""
function repeatcoord(coordinates, values)
    vcat(([coord for _ in 1:v] for (coord, v) in zip(coordinates, values))...)
end

"""
    kde2d((xs, ys), values)

Performs a 2d KDE based on two lists of coordinates, and the value at those coordinates.
"""
function kde2d((xs, ys), values)
    xsr, ysr = [repeatcoord(coord, values) for coord in (xs, ys)]
    coords = hcat(xsr, ysr)
    f = kde(coords)
    return f
end

# Load Data

data = load("transcriptomics_data.jld")
coordinates = data["coordinates"]
M = data["count_matrix"]

m,n = 4180,4021 # number of cells x number of genes (features)

# Each row of M is the values for a gene
# Ex. M[1,:] gives the values for the first gene
# The different columns correspond to the coordinates the genes are sampled at:

xs, ys = coordinates[1,:], -coordinates[2,:]

begin
gene = 1 #181
# Plot the coordinates with darker points indicating larger values,
scatter(xs, ys, markeralpha=M[gene,:] ./ maximum(M[gene,:])) |> display

f = kde2d((xs, ys), M[gene,:])

heatmap(f.x, f.y, f.density) |> display
end

# Extract all genes and compile into a tensor
n_genes = 4021 #n
J = K = 2^5 # Number of samples in each dimention
I = n_genes

Y = zeros(I, J, K) # Data tensor

xs_resample = range(f.x[begin], f.x[end], length=J)
ys_resample = range(f.y[begin], f.y[end], length=K)

for gene in 1:n_genes #number of genes
    @show gene
    f = kde2d((xs, ys), M[gene,:])
    Y[gene, :, :] = pdf(f, xs_resample, ys_resample)
end

heatmap(xs_resample, ys_resample, Y[1, :, :]) |> display

# Normalize the sum of each gene (horizontal) slice
Y_slices = eachslice(Y, dims=1)
slice_sums = sum.(Y_slices)
Y_slices ./= slice_sums

###########

# Decomposition
R = 12

Random.seed!(314) #to get the same initialization

fact = BlockTensorFactorization.Core.factorize
decomposition, stats, kwargs = fact(Y; rank=12, stats=[RelativeError,GradientNorm,GradientNNCone], converged=[GradientNNCone],tolerance=1e-5, maxiter=1000, constraints=[l1scale_1slices! ∘ nonnegative!, nonnegative!]);

rel_errors, norm_grad, dist_Ncone = stats[:,:RelativeError], stats[:,:GradientNorm], stats[:,:GradientNNCone]

F, C = factors(decomposition)

plot(rel_errors,yaxis=:log10) |> display
plot(norm_grad,yaxis=:log10) |> display
plot(dist_Ncone,yaxis=:log10) |> display

Y_hat = mtt(C,F)

rel_error(x, x_true) = norm(x - x_true) / norm(x_true)

@show rel_error(Y,Y_hat)

mean_rel_error = mean(rel_error.(eachslice(Y;dims=1), eachslice(Y_hat;dims=1)))

@show mean_rel_error

p = heatmap(C; xaxis="cell type", yaxis="gene", xticks=1:12, size=(500,300),yflip=true, yticks=0:500:4000,clims=(0,1))
display(p)
# savefig(p, "../figures/transcriptomics/gene_profiles.svg")

for r in 1:R
    p = heatmap(F[r,:,:], title="cell type $r", xticks=false,yticks=false,colorbar=false,titlefontsize=18)
    display(p)
    # savefig(p, "../figures/transcriptomics/celltype$r.svg")
end
