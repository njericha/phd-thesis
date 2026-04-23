# bass_flute.jl
#
# Script to separate an audio clip of a bass and flute played simultaneously

using Random
using LinearAlgebra
using Statistics: mean

using JLD
using Plots
using Interpolations
using StatsBase
using STFT
using WAV

using BlockTensorFactorization

Random.seed!(314)

# Data loading
# This file already contains the STFT of the flute-guitar audio clip with info about the time and frequency points
d = load("real_music_data.jld")
Y, Φ, Xs, xs, D, DD, times, midi_notes, freqs =
    d["Y"], d["Φ"], d["Xs"], d["xs"], d["D"], d["DD"], d["times"], d["midi_notes"], d["freqs"]

full_time_samples = 0.0:0.000181440111620479:1.853410740203193
sample_rate = 5512.0

input_mixture, _ = wavread("flute_and_guitar.wav");
downsample = 8
input_mixture = input_mixture[1:downsample:N]
input_mixture ./= maximum(abs.(input_mixture))
plot(range(0,full_time_samples[end];length=length(input_mixture)),input_mixture;
    legend = false,
    xlabel="time (s)",
    ylabel="normalized amplitude",
    size=(500,300)
) |> display

# Check the input STFT
heatmap(times, freqs, Y;
    xlabel="time (s)",
    ylabel="frequency (Hz)",
    clims=(0,15),
    size=(500,300)
) |> display

# Factorize
R = 10
fact = BlockTensorFactorization.factorize
decomposition, stats, kwargs = fact(Y';
    rank=R,
    model=Tucker1,
    constraints=[linftyscale_rows! ∘ nonnegative!, nonnegative!],
    converged=RelativeError,
    maxiter=2000,
    tolerance=0.085 # Relative error less than 8.5%
)
display(stats)

B, A = factors(decomposition)
dist_Ncone = stats[:,:GradientNNCone]

heatmap(A') |> display
heatmap(B) |> display
heatmap(Y; title="input") |> display

heatmap((A*B)'; title="learned") |> display

plot(dist_Ncone; yscale=:log10) |> display

######### Harmonic Error ##########
plot(A[:,2])
start_indx = 4
plot(freqs[start_indx:end],B[:,start_indx:end]'; xscale=:log10)
bs = collect.(eachrow(B))

v = bs[4]
plot(freqs,v) |> display

# Interpolate v on an exponential grid
# Conditions are
# 1) start at the first non-zero freq
# 2) the last two samples should match freqs[end-1], freqs[end]
@show freqs[end-1], freqs[end]

a,b,c = freqs[2], freqs[end-1], freqs[end]

"""
    exp_samples(a,b,c)

Makes a an exponential spaces domain of samples where the following holds:
`domain[1] == a`
`domain[N-1] == b`
`domain[N] == c`.

Assumes none of a,b,c are 0, and a < b < c.
"""
function exp_samples(a,b,c)
    N = floor(Integer,log(c/a)/log(c/b) + 1)
    A = a*b/c
    C = log(c/b)

    i = 1:N
    domain = @. A*exp(C*i)
    return domain
end

# Make log spaced frequencies
log_spaced_freqs = exp_samples(a,b,c)

interp = Interpolations.interpolate

# Make Interpolate object from the spectrum v which is the values at points lin_spaced_freqs
lin_spaced_freqs = (collect(freqs),)
interp_spectrum4 = interp(lin_spaced_freqs, v, Gridded(Linear()))

# Evaluate the interpolation at the new points log_spaced_freqs
log_spaced_spectrum4 = interp_spectrum4(log_spaced_freqs)

# Check the interpolation matches the original!
plot(log_spaced_freqs,log_spaced_spectrum4)
plot!(freqs,v)

# Now do it for every spectrum
log_spaced_spectrums = [interp(lin_spaced_freqs, v, Gridded(Linear()))(log_spaced_freqs) for v in bs]

log_spaced_matrix = hcat(log_spaced_spectrums...)

# Compute cross-correlation between every spectrum (columns of the matrix)
# the 3d order array CC holds the correlation for lag i between columns j and k in CC[i,j,k]
# the 1-vectors CC[:, j, k] hold the complete correlation for all lags
n = length(freqs)
CC = crosscor(log_spaced_matrix, log_spaced_matrix, -n+1:n-1; demean=false)

plot(-n+1:n-1,CC[:,2,5]) |> display

plot(log_spaced_matrix[:,[2,5]]) |> display

# Find the maximum cross-correlation value  for each pair
# we don't care about at what lag/frequency shift this occurs
max_CC = reshape(maximum(CC; dims=1), size(CC)[2:3])
plot_pre_threshold = heatmap(max_CC;aspect_ratio=1,xticks=1:10,yticks=1:10,size=(400,400),yflip=true,colorbar=false)
display(plot_pre_threshold)

"""
    cluster(W::Symmetric, t)

Cluster nodes based on their (undirected) edge weights `W`.

The threshold `t` gives a thresholding parameter which will only link nodes with
edges bigger than `t` (recursively).

Example 1
-------
julia> W = [
1.0       0.493897  0.856425  0.870541  0.529691;
0.493897  1.0       0.689449  0.640682  0.962191;
0.856425  0.689449  1.0       0.979603  0.728575;
0.870541  0.640682  0.979603  1.0       0.702165;
0.529691  0.962191  0.728575  0.702165  1.0;
]

julia> t = 0.9

julia> cluster(W, t) == Set((
    Set((2,5)),
    Set((4,3)),
    Set((1)),
))

Example 2
-------
julia> W = [
1 0 1 0 0;
0 1 0 1 0;
1 0 1 0 1;
0 1 0 1 1;
0 0 1 1 1;
]

julia> t = 1

julia> cluster(W, t) == Set(Set([1, 2, 3, 4, 5]))
"""
function cluster(W, t)
    issymmetric(W) || throw(ArgumentError("W should be symmetric, got $W"))

    A = W .>= t # Adjacency matrix

    # Start by creating all neighbourhoods
    # The "findall" finds the edges (where the elements are 1)
    neighbourhoods = Set(Set(findall(row)) for row in eachrow(A))

    for n in neighbourhoods
        for m in neighbourhoods
            if (n === m) || isempty(n ∩ m)
                # skip neighbourhoods that are the same (the code in the else statement would do nothing)
                # or neighbourhoods with no overlap
                continue
            else # Merge n and m since there is overlap
                # Remove n and m from the set of neighbourhoods
                setdiff!(neighbourhoods, (n, m))

                # Add n ∪ m to the set of neighbourhoods
                push!(neighbourhoods, n ∪ m)
            end
        end
    end

    return neighbourhoods
end

cluster_threshold = 0.7
clusters = cluster(max_CC, cluster_threshold)

plot_post_threshold = heatmap(max_CC .>= cluster_threshold;aspect_ratio=1,xticks=1:10,yticks=1:10,size=(400,400),yflip=true,colorbar=false)
display(plot_post_threshold)

plot_colour_bar = scatter([0], [1];
        zcolor=[0,1],
        clims=(0,1),
        xlims=(1,1.01),
        xticks=false,
        yticks=false,
        xshowaxis=false,
        yshowaxis=false,
        markerfill=false,
        label="",
        colorbar=:left,
        aspect_ratio=10,
        size=(70,400)
        )

## Plot before and after threshold in one plot
l = @layout [a{0.4w} b{0.4w} c{0.2w}]
plot(
    heatmap(max_CC; aspect_ratio=1,xticks=1:10,yticks=1:10, colorbar=false),
    heatmap(max_CC .>= cluster_threshold;aspect_ratio=1,xticks=1:10,yticks=1:10,colorbar=false),
    scatter([0], [1];
        zcolor=[0,1],
        clims=(0,1),
        xlims=(1,1.01),
        xticks=false,
        yticks=false,
        xshowaxis=false,
        yshowaxis=false,
        markerfill=false,
        label="",
        colorbar=:left,
        aspect_ratio=10
        );
    layout=l, grid=false,yflip=true, size=(1035,400)
) |> display

"""
    regroup(A, B, clusters)

Groups columns of A and rows of B according to the clusters.

Each returned spectrum i is given by ∑_(r ∈ Cᵢ) A[:, r]*(B[r, :])'

Returns
-------
`spectrums::Vector{Matrix}`
"""
function regroup(A,B, clusters)
    spectrums = Vector{Matrix{typeof(A[begin]*B[begin])}}(undef, length(clusters))
    for (i, C) in enumerate(clusters)
        indx = collect(C)
        @show A[:,indx]
        @show B[indx, :]
        spectrum = (@view A[:,indx]) * (@view B[indx, :])
        spectrums[i] = spectrum
    end
    return spectrums
end

spectrums = regroup(A,B, clusters)

for spectrum in spectrums
    heatmap(times, freqs, spectrum';
        xlabel="time (s)",
        ylabel="frequency (Hz)",
        size=(500,300),
        clims=(0,15),
    ) |> display
end

################################
### Recovery to Time Domain ####
################################

# signal processing stuff

## STFT
"""The Hann Windowing function"""
function hann(N::Integer)
    N = N - N % 2 # makes sure N even
    n = 0:N
    return @. sin(π*n/N)^2
end

window_width = 300
hop = window_width÷2 - 1 # number of samples to hop over
window = hann(window_width)
mystft(y) = stft(y, window, hop)
myistft(Y) = istft(Y, window, hop)
freq_to_time(X;chopsize=20) = myistft(X .* cis.(Φ))[begin+chopsize:end-chopsize]

# Take the iSTFT to get time domain sources
learned_sources = [freq_to_time(spectrum') for spectrum in spectrums]

# Get cropped time domain samples
choppedlength = length(full_time_samples) - length(learned_sources[begin])
crop(x;choppedlength=choppedlength) = x[begin + (choppedlength+1)÷2 : end - choppedlength÷2]
tchop = crop(full_time_samples)

for source in learned_sources
    plot(tchop, source ./ maximum(abs.(source));
        legend = false,
        xlabel = "time (s)",
        ylabel= "normalized amplitude",
        ylims=(-1,1),
        size=(500,300)
    ) |> display
end

for source in learned_sources
    wavplay(source, sample_rate)
end

guitar = xs[1]
flute = xs[2]

guitar_theoretical_best = crop(guitar)
flute_theoretical_best = crop(flute)

plot(guitar_theoretical_best) |> display
plot(flute_theoretical_best) |> display

source2, source1 = learned_sources

for source in [guitar_theoretical_best,flute_theoretical_best]
    wavplay(source, sample_rate)
end

wavwrite(source1, "learned_guitar.wav", Fs=sample_rate)
wavwrite(source2, "learned_flute.wav", Fs=sample_rate)
