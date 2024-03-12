# Some helper functions to do large numbers of Larter-Breakspear simulations

module generateLotsaLarter

export makeSobolSeqParams, checkForExpansion, filterTimeSeries

using Sobol, Plots, DSP, Statistics

c = [0.2, 0.5] #Coupling constant
δ = [0.64, 0.7] #δ_v,z
gca = [0.95, 1.05] #Ca conductance
eca = [0.95, 1.01] #Ca Nernst potential
gk = [1.95, 2.05] #K conductance
ek = [-0.75, -0.65] #K Nernst potential
gna = [6.6, 6.8] #Na conductance
ena = [0.48, 0.58] #Na Nernst potential
aee = [0.33, 0.39] #excitatory -> excitatory connection strength
aei = [1.95, 2.05] #excitatory -> inhibitory connection strength
nmda = [0.20, 0.30] #NMDA/AMPA receptor ratio

"""
    makeSobolSeqParams(numPoints, skipFactor)

Generate Sobol sequence (quasi-random low discrepancy sequence on a hypercube)
Remember: the sequence isn't random! So if you're doing multiple runs, you'll need to use the skip appropriately to get to unsampled values in the space
"""
function makeSobolSeqParams(numPoints, skipFactor)
    s = SobolSeq([c[1], δ[1], gca[1], eca[1], gk[1], ek[1], gna[1], ena[1], aee[1], aei[1], nmda[1]], [c[2], δ[2], gca[2], eca[2], gk[2], ek[2], gna[2], ena[2], aee[2], aei[2], nmda[2]])
    s = skip(s, numPoints * skipFactor, exact=true)
    p = reduce(hcat, next!(s) for i = 1:numPoints)'
    return p
end

"""
    checkForExpansion(timeseries)

Check if time series is expanding out of bounds (bad initial condition or parameter choice)
"""
function checkForExpansion(timeseries)
    expanding = false
    for i in 1:size(timeseries, 1)
        if length(findall(x -> x > 1.5, timeseries[i, :])) != 0 || length(findall(x -> x < -1.5, timeseries[i, :])) != 0
            expanding = true
        end
    end
    return expanding
end

"""
    checkForNoOscillation(timeseries, tol)

Check if time series doesn't oscillate (based on low standard deviation)
"""
function checkForNoOscillation(timeseries, tol)
    noOscillation = false
    for i in 1:size(timeseries, 1)
        if std(timeseries[i, :]) < tol
            noOscillation = true
        end
    end
    return noOscillation
end

"""
    checkForUniformOscillation(cc_matr)

Check if correlation matrix is uniform (or close to it)
"""
function checkForUniformOscillation(cc_matr)
    uniform = true
    if mean(cc_matr) < 0.985
        uniform = false
    end
    return uniform
end

"""
    filterTimeSeries(timeseries)

Filter time series to match data, which is a 4th-order Butterworth bandpass filter with
frequency: [0.01, 0.1] Hz
TR: 0.8s -> sampling frequency 1.25Hz
"""
function filterTimeSeries(timeseries)
    responsetype = Bandpass(0.01, 0.1; fs=1.25)
    designmethod = Butterworth(4)
    filtered_ts = zeros(size(timeseries))
    for i in 1:size(timeseries, 1)
        filtered_ts[i, :] = filt(digitalfilter(responsetype, designmethod), timeseries[i, :])
    end
    return filtered_ts
end

end