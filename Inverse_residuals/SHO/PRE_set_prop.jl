using LazySets, Plots, IntervalArithmetic
import IntervalArithmetic as IA
using NPZ
using FFTW

## Some params
m = 1
k = 1
D_tt_kernel = [1, -2, 1] 
dt = 0.1010101
D_identity = [0, 1, 0]


# Cell 5: Define inverse kernel function
function compute_inverse(kernel_fft, eps=1e-16)
    return 1 ./ (kernel_fft .+ eps)
end

include("intervalFFT.jl")

numerical_sol = npzread("ODE_outputs.npy")
neural_sol = npzread("Nueral_outputs.npy")

function set_PRE(neural_test)

    D_pos_kernel = m*D_tt_kernel + dt^2*k*D_identity

    signal_padded = [0; neural_test[:, 1]; 0]

    N_signal = size(signal_padded, 1)

    N_pad = N_signal - length(D_pos_kernel)
    kernel_pad = vcat(D_pos_kernel, zeros(N_pad))

    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)

    convolved = ifft(signal_fft .* kernel_fft)
    inverse_kernel = compute_inverse(kernel_fft)

    convolved_noedges = convolved[4:end-1]
    right_edges = convolved[1:3]
    left_edges = convolved[end]

    # convolved_set_center = interval.(min.(real(convolved_noedges), 0), max.(real(convolved_noedges), 0))
    convolved_set_center = interval.(-abs.(real(convolved_noedges)), abs.(real(convolved_noedges)))
    convolved_set_right = interval.(real(right_edges))
    convolved_set_left = interval.(real(left_edges))

    convolved_set = [convolved_set_right; convolved_set_center; convolved_set_left]

    # Begin set stuff
    # convolved_set = interval.(min.(real(convolved), 0), max.(real(convolved), 0))
    # convolved_set = interval.(-abs.(real(convolved)), abs.(real(convolved)))

    convolved_set_fft = intervalFFT(convolved_set)

    convolved_set_fft_kernel = complex_prod.(convolved_set_fft, inverse_kernel)

    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)

    return Real.(retrieved_signal)
end


ID = rand(1:300)
neural_test = neural_sol[ID, :, 1]
numerical_test = numerical_sol[ID, :, 1]

signal_bounds = set_PRE(neural_test)

signal_bounds_back = signal_bounds[2:end-1]

is_it_in_numerical = all(numerical_test[:, 1] .∈ signal_bounds_back)
is_it_in_neural = all(neural_test[:, 1] .∈ signal_bounds_back)

println("Numerical is in: $is_it_in_numerical and Neural is in: $is_it_in_neural")

plot(neural_test[:, 1], label = "neural")
plot!(numerical_test[:, 1], label = "numerical")
plot!(sup.(signal_bounds_back), fill_between = inf.(signal_bounds_back), alpha = 0.2)