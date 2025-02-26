using LazySets, Plots, IntervalArithmetic
import IntervalArithmetic as IA

###
# Set propagation through discrete FFT and inverses. 
# 
# We treat a set of complex numbers as a zonotope (https://polytope.miraheze.org/wiki/Zonotope)
# with the X1 dimension corresponding to the real components, and X2 as the imaginary.
# The FFT's linearity means set propagation is efficient with zonotopes, and preserves zonotope structure 
# (i.e. we don't need anything more complex).
#
# The product of a zonotopic complex number and a precise complex number corresponds to a scaling and rotation.
###


function complex_prod(Z::Zonotope, C::ComplexF64)

    scaling_fac = abs(C)
    angle = atan(C.im, C.re)

    rot_matrix = [cos(angle) -sin(angle); sin(angle) cos(angle)]

    Z_rot = overapproximate(rot_matrix * Z, Zonotope)
    return scale(scaling_fac, Z_rot)
end

function test_complex_prod()

    Z = Zonotope(randn(2), randn(2,2))
    C = Complex(randn(), randn())

    Z_rot_scale = complex_prod(Z, C)

    plot(Z)
    plot!(Z_rot_scale)
    plot!(Singleton([C.re; C.im]))

    Nsim = 500
    Z_rand = sample(Z, Nsim)
    Z_rand_C = [Complex(Z_r...) for Z_r in Z_rand]

    Z_rot_C = Z_rand_C .* C

    scatter!(Z_rot_C)

end

function intervalFFT_(Xk :: Vector{IA.Interval{Float64}}, h)

    N_data = length(Xk)

    ks = 0:(N_data-1)
    thetas = 2*π/N_data .* ks * h

    rot_matrix = [cos.(thetas) 0 .- sin.(thetas)]
    Xk_lazy = convert.(LazySets.Interval, Xk)
    Zk = overapproximate.(Xk_lazy, Zonotope)

    Zk_rot = [rot_matrix[i,:] * Zk[i] for i = 1:N_data]
    Zs = overapproximate.(Zk_rot, Zonotope)
    
    Z_out = minkowski_sum(Zs[2], Zs[1])

    for i = 3:N_data
        Z_out = minkowski_sum(Zs[i], Z_out)
    end
    return Z_out
end

function inverse_intervalFFT_(Zh :: Vector{Zonotope{N, S, T}}, k) where {N, S, T}

    N_data = length(Zh)

    hs = 0:(N_data-1)
    thetas = 2*π/N_data .* hs * k

    rot_matrix = [[cos(θ) 0 .- sin(θ); sin(θ) cos(θ)] for θ in thetas]

    Zh_rot = [rot_matrix[i] * Zh[i] for i = 1:N_data]
    Zs = overapproximate.(Zh_rot, Zonotope)

    Z_out = minkowski_sum(Zs[2], Zs[1])

    for i = 3:N_data
        Z_out = minkowski_sum(Zs[i], Z_out)
    end
    # return overapproximate((1/N_data) * Z_out, Zonotope)
    return scale(1/N_data, Z_out)
end

function intervalFFT(Xk :: Vector{IA.Interval{Float64}})
    return [intervalFFT_(Xk, i) for i = 0:length(Xk)-1]
end

function inverse_intervalFFT(Zh :: Vector{Zonotope{N, S, T}}) where {N, S, T}
    return [inverse_intervalFFT_(Zh, i) for i = 0:length(Zh)-1]
end

###
#   There's a mistake in the lower bound in the 2nd case. You need to check the midpoint between the 
#   2 lowest amplitudes
###
function amplitute(Z :: Zonotope)
    amplitudes = norm.(vertices(Z))
    if in(zeros(2), Z)
        return interval( 0, maximum(amplitudes) )
    end
    return interval(minimum(amplitudes), maximum(amplitudes))
end

function amplitute(Z::VPolygon)
    amplitudes = norm.(Z.vertices)
    if in(zeros(2), Z)
        return interval( 0, maximum(amplitudes) )
    end
    return interval(minimum(amplitudes), maximum(amplitudes))
end

function Real(Z :: Zonotope)
    Z_highs = high(Z)
    Z_los = low(Z)

    return interval.(Z_los[1], Z_highs[1])
end

function box(Z :: Zonotope)

    Z_highs = high(Z)
    Z_los = low(Z)

    Z_real = interval.(Z_los[1], Z_highs[1])
    Z_Complex = interval.(Z_los[2], Z_highs[2])

    return Z_real, Z_Complex
end

function box(Z :: VPolygon)

    Z_highs = high(Z)
    Z_los = low(Z)

    Z_real = interval.(Z_los[1], Z_highs[1])
    Z_Complex = interval.(Z_los[2], Z_highs[2])

    return Z_real, Z_Complex
end

