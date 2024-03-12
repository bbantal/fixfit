# DiffEq for solvers, MTK for symbolic expression, Sobol for sobol sequence generation, MAT for saving to .mat files
using DifferentialEquations, ModelingToolkit, Sobol, MAT

# This is the MTK 8 way of doing things - use t_nounits and D_nounits in MTK 9
@variables t
D = Differential(t)

n_sim = 10000 # use this first to skip the required number of initial steps in Sobol

# Parameters that vary across simulations
C_orig = 0.001
Sᵢ_orig = 0.0005
p_orig = 0.03
α_orig = 7.85
γ_orig = 0.3
meal_bolus_orig = 4

# For all non-bolus parameters, scale by an order of magnitude up and down
# For bolus, scale within normal physiological range

lb = [C_orig, Sᵢ_orig, p_orig, α_orig, γ_orig, 2.0] .* [0.1, 0.1, 0.1, 0.1, 0.1, 1.0]
ub = [C_orig, Sᵢ_orig, p_orig, α_orig, γ_orig, 5.0] .* [10.0, 10.0, 10.0, 10.0, 10.0, 1.0]

s = SobolSeq(lb, ub)
s = skip(s, n_sim) # skip the first steps to get a better distribution - see Sobol.jl documentation

for run = 1:n_sim
    params_new = next!(s) # get next parameter set from sequence generator
    
    # Equations are from Karin et al. 2016. Initial conditions are set to be at a stable point.
    @parameters C=params_new[1] Sᵢ=params_new[2] p=params_new[3] α=params_new[4] γ=params_new[5] G₀=5.4 β₀=322.0 μ₊=0.000014583 μ₋=0.000017361
    @variables G(t)=G₀ I(t)=p*β₀*(G₀^2/(α^2+G₀^2))/γ β(t)=β₀
    λ₊(G) = μ₊*1/(1+((8.4/G)^1.7))
    λ₋(G) = μ₋*1/(1+((G/4.8)^8.5))
    eqs = [D(G) ~ G₀*(C+Sᵢ*(p*β₀*(G₀^2/(α^2+G₀^2))/γ)) - (C+Sᵢ*I)*G,
       D(I) ~ p*β*(1/(1+(α/G)^2))-γ*I,
       D(β) ~ β*(λ₊(G) - λ₋(G))]

    t₁ = 9*60 # Breakfast at 9am
    t₂ = 13*60 # Lunch at 1pm
    t₃ = 18*60 # Dinner at 6pm
    meal_bolus = params_new[6]
    meals = ((t == t₁) | (t == t₂) | (t == t₃)) => [G ~ G + meal_bolus]

    @named sys = ODESystem(eqs, t; discrete_events=meals)
    prob = ODEProblem(sys, [], (0.0, 1440.0))
    # Try is just to make sure that the parameters didn't result in a degenerate condition - some choices can lead to complex solutions.
    try 
        sol = solve(prob, Tsit5(); tstops=[t₁, t₂, t₃], saveat=1.0)
        if sol.retcode == ReturnCode.Success
            file = matopen("/Users/achesebro/Downloads/sim_results/test_sobol"*lpad(run, 5, "0")*".mat", "w")
            write(file, "params", params_new)
            write(file, "sol", convert(Array, sol))
        end
    catch
        println("Error in run "*lpad(run, 5, "0"))
    end
end