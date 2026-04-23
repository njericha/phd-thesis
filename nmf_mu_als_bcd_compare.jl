# nmf_mu_als_bcd_compare.jl
#
# Compare three algorithms for computing a low nonnegative rank matrix factorization
# MU: Multiplicative Updates
# ALS: Alternating Least-Squares
# BCD: Block Coordinate Descent

using Random
using NMF

using LinearAlgebra
using Plots
using IOCapture
using DataFrames
using Statistics

Random.seed!(3141592653589793238462643)


#################
# Generate Data #
#################

I, J = 200, 200
R = 5
A = abs.(randn(I, R))
B = abs.(randn(R, J))
Y = A * B

##################
# Run Algorithms #
##################

tol = 1e-16
maxiter_total = 10000
maxsubiter = 50
maxiter = maxiter_total ÷ maxsubiter
verbose = true
trials = 20
new_init_each_trial = true

A_init = abs.(randn(I, R))
B_init = abs.(randn(R, J))

MU = NMF.MultUpdate{Float64}(;tol,maxiter=maxiter_total,verbose);
ALS = NMF.ALSPGrad{Float64}(;tol,maxiter=maxiter*8,maxsubiter,verbose);
BCD = NMF.ALSPGrad{Float64}(;maxsubiter=1,tol,maxiter=maxiter_total,verbose);

function parse_output(string_out)
    # Split rows
    string_out = split(string_out, "\n"; keepempty=false)
    # Split Columns
    string_out = split.(string_out, "  "; keepempty=false)
    # Clean up spacing
    remove_leading_space(str) = str[1] == ' ' ? str[2:end] : str
    string_out = map.(remove_leading_space, string_out)
    # Extract headers and data
    cols = 2:3
    headers = string_out[1][cols]
    data_rows = map(x -> parse.(Float64,x[cols]),string_out[2:end])
    # Format as a DataFrame
    output = DataFrame(hcat(data_rows...)',headers) # DataFrames expects column data
    return output
end

MU_outs = DataFrame[]
ALS_outs = DataFrame[]
BCD_outs = DataFrame[]

# Need these copies outside the for loop for scoping
A_MU = copy(A_init)
B_MU = copy(B_init)

A_ALS = copy(A_init)
B_ALS = copy(B_init)

A_BCD = copy(A_init)
B_BCD = copy(B_init)

# run once to initialize the function
MU_out = IOCapture.capture() do
    NMF.solve!(MU, Y, A_MU, B_MU)
end;

for trial in 1:trials
    println("Trial ",trial," of ",trials)

    if new_init_each_trial
        A_init = abs.(randn(I, R))
        B_init = abs.(randn(R, J))
    end

    A_MU .= copy(A_init)
    B_MU .= copy(B_init)

    A_ALS .= copy(A_init)
    B_ALS .= copy(B_init)

    A_BCD .= copy(A_init)
    B_BCD .= copy(B_init)

    MU_out = IOCapture.capture() do
    NMF.solve!(MU, Y, A_MU, B_MU)
    end;

    ALS_out = IOCapture.capture() do
    NMF.solve!(ALS, Y, A_ALS, B_ALS)
    end;

    BCD_out = IOCapture.capture() do
    NMF.solve!(BCD, Y, A_BCD, B_BCD)
    end;

    push!(MU_outs, parse_output(MU_out.output))
    push!(ALS_outs, parse_output(ALS_out.output))
    push!(BCD_outs, parse_output(BCD_out.output))
end

# We use the same initialization each trial so the output should be the same
# There is only variance in the time taken
X_MU = A_MU * B_MU
X_ALS = A_ALS * B_ALS
X_BCD = A_BCD * B_BCD

###########
# Results #
###########

rel_error(approx, truth) = norm(approx - truth) / norm(truth)

@show rel_error(X_MU, Y)
@show rel_error(X_ALS, Y)
@show rel_error(X_BCD, Y)

@show norm(A_MU)
@show norm(A_ALS)
@show norm(A_BCD)

@show norm(B_MU)
@show norm(B_ALS)
@show norm(B_BCD)

extract_times(x) = eachrow(hcat(map(x -> x[!,"Elapsed time"], x)...))
extract_objective(x) = x[1][!,"objv"] # all trials are the same
top_quantile = 0.95
bot_quantile = 0.05

MU_times = extract_times(MU_outs)
ALS_times = extract_times(ALS_outs)
BCD_times = extract_times(BCD_outs)

MU_times_top = quantile.(MU_times, top_quantile)
ALS_times_top = quantile.(ALS_times, top_quantile)
BCD_times_top = quantile.(BCD_times, top_quantile)

MU_times_bot = quantile.(MU_times, bot_quantile)
ALS_times_bot = quantile.(ALS_times, bot_quantile)
BCD_times_bot = quantile.(BCD_times, bot_quantile)

MU_times_median = median.(MU_times)
ALS_times_median = median.(ALS_times)
BCD_times_median = median.(BCD_times)

MU_obj = extract_objective(MU_outs)
ALS_obj = extract_objective(ALS_outs)
BCD_obj = extract_objective(BCD_outs)
begin
p = plot(;
    xlabel="time (s)",
    ylabel="objective",
    # xticks=(problem_sizes .- 1),
    yticks=[10. ^n for n in -12:3:5],
    # xaxis=:log10,
    yaxis=:log10,
    xlims=(0,0.5),
    ylims=(10.0^(-13), 10.0^5),
    size=(450,250),
    legend=:bottomleft,
    fontfamily="Computer Modern",
    )
function plot_xbounds(p, top,bot,median,vals;color,kwargs...)
    xs, ys = [bot; reverse(top)], [vals; reverse(vals)]
    plot!(p, xs, ys;
    st=:shape,
    fc=color,
    alpha=0.2,
    lw=0,
    label=false,
    )
    plot!(p, median, vals; color,kwargs...)
end
plot_xbounds(p, MU_times_top,MU_times_bot,MU_times_median,MU_obj;color=:blue,label="MU",linewidth=2,linestyle=:dash)
plot_xbounds(p, ALS_times_top,ALS_times_bot,ALS_times_median,ALS_obj;color=:orange,label="ALS",linewidth=2,linestyle=:dot)
plot_xbounds(p, BCD_times_top,BCD_times_bot,BCD_times_median,BCD_obj;color=:green,label="BCD",linewidth=2,linestyle=:dashdot)
end
