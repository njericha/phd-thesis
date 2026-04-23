# subblock_momentum_benchmark.jl
#
# Benchmarks two features of BlockTensorFactorization.jl: Subblock Descent and Momentum

using BenchmarkTools
using Logging
using BlockTensorFactorization
using Random
using Statistics

Random.seed!(3141592653589)

global_logger(SimpleLogger(Warn))

fact = BlockTensorFactorization.factorize

R = 3
options = (
    :rank => R,
    :tolerance => (0.05),
    :converged => (RelativeError),
    :δ => 0.999,
    :maxiter => 500,
    :model => CPDecomposition,
    :stats => [Iteration, RelativeError],
)

n_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=false,
    options...
)

y_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=false,
    options...
)

n_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=true,
    options...
)

y_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=true,
    options...
)

performance_increase(old, new) = (old - new) / new * 100

time_decrease(old, new) = (old - new) / old * 100

I, J, K = 10, 10, 10
Y = CPDecomposition((I, J, K), R) |> array

#run once to compile
decomposition, stats, kwargs = n_subblock_n_momentum(Y)
decomposition, stats, kwargs = y_subblock_n_momentum(Y)
decomposition, stats, kwargs = n_subblock_y_momentum(Y)
decomposition, stats, kwargs = y_subblock_y_momentum(Y)

trials = 100
n_subblock_n_momentum_times = zeros(trials)
n_subblock_y_momentum_times = zeros(trials)
y_subblock_n_momentum_times = zeros(trials)
y_subblock_y_momentum_times = zeros(trials)

n_subblock_n_momentum_iterations = zeros(trials)
n_subblock_y_momentum_iterations = zeros(trials)
y_subblock_n_momentum_iterations = zeros(trials)
y_subblock_y_momentum_iterations = zeros(trials)

for trial in 1:trials
    println("Trial ", trial, " of ", trials)
    Y=CPDecomposition((I, J, K), R)|>array

    out = @timed n_subblock_n_momentum(Y)
    n_subblock_n_momentum_times[trial] = out.time
    n_subblock_n_momentum_iterations[trial] = out.value[2][end,:Iteration]

    out = @timed n_subblock_y_momentum(Y)
    n_subblock_y_momentum_times[trial] = out.time
    n_subblock_y_momentum_iterations[trial] = out.value[2][end,:Iteration]

    out = @timed y_subblock_n_momentum(Y)
    y_subblock_n_momentum_times[trial] = out.time
    y_subblock_n_momentum_iterations[trial] = out.value[2][end,:Iteration]

    out = @timed y_subblock_y_momentum(Y)
    y_subblock_y_momentum_times[trial] = out.time
    y_subblock_y_momentum_iterations[trial] = out.value[2][end,:Iteration]
end

experiment_times = (n_subblock_n_momentum_times, n_subblock_y_momentum_times, y_subblock_n_momentum_times, y_subblock_y_momentum_times)

experiment_iterations = (n_subblock_n_momentum_iterations, n_subblock_y_momentum_iterations, y_subblock_n_momentum_iterations, y_subblock_y_momentum_iterations)

@show median.(experiment_times)
@show median.(experiment_iterations)

median_iterations = median.(experiment_iterations)
median_times = median.(experiment_times)

base_iterations = median_iterations[1]

base_time = median_times[1]

for n_iterations in median_iterations
    @show performance_increase(base_iterations, n_iterations)
end

for time in median_times
    @show time_decrease(base_time, time)
end
