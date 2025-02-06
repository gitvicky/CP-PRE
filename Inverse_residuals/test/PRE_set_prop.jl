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

    N_signal = size(neural_test,1)

    N_pad = N_signal - length(D_pos_kernel)
    kernel_pad = vcat(D_pos_kernel, zeros(N_pad))

    signal_fft = fft(neural_test[:, 1])
    kernel_fft = fft(kernel_pad)

    convolved = ifft(signal_fft .* kernel_fft)
    inverse_kernel = compute_inverse(kernel_fft)

    # Begin set stuff
    convolved_set = interval.(min.(real(convolved), 0), max.(real(convolved), 0))
    # convolved_set = interval.(-abs.(real(convolved)), abs.(real(convolved)))

    convolved_set_fft = intervalFFT(convolved_set)

    convolved_set_fft_kernel = complex_prod.(convolved_set_fft, inverse_kernel)

    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)

    return Real.(retrieved_signal)
end


ID = rand(1:500)
neural_test = neural_sol[ID, :, :]
numerical_test = numerical_sol[ID, :, :]

signal_bounds = set_PRE(neural_test)

plot(sup.(signal_bounds), fill_between = inf.(signal_bounds), alpha = 0.2)
plot!(neural_test[:, 1], label = "neural")
plot!(numerical_test[:, 1], label = "numerical")