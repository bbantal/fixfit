# Actively under construction
# Ask Anthony is you need to run this (although no pubs depend on it yet, so probably unnecessary)

using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, CSV, DataFrames,LinearAlgebra, MAT, Random, Plots, Statistics

include("fmri_ac.jl")
include("generateLotsaLarter.jl")
savepath = "/shared/datasets/private/LB_model_outputs/botond_parameter_fitting/bonus_c_and_na_runs/"

connLB = matread(joinpath(@__DIR__, "DATA","mean_structural_connectivity.mat"))
clb = connLB["mean_structural_connectivity"]

function LinearConnectionsLB(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
		norm_factor = sum(adj_matrix[:, region_num])
		# @show num_conn
    	push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num])/norm_factor)
    end
    return @named Circuit = ODESystem(eqs, systems = sys)
end

blox = []
for i = 1:78
	lb = LarterBreakspearBlox(name=Symbol("LB$i"))
	push!(blox,lb)
end

sys = [b.odesystem for b in blox]
con = [s.connector for s in blox]
@named LB_circuit_lin = LinearConnectionsLB(sys=sys, adj_matrix=clb,connector=con)
mysys = structural_simplify(LB_circuit_lin)

mysys.states
mysys.ps

uw = rand(Int(length(mysys.states)/3))
uv = 0.1*rand(Int(length(mysys.states)/3)) .- 0.05
uz = 0.1*rand(Int(length(mysys.states)/3)) .- 0.05
u0 = collect(Iterators.flatten(zip(uv,uz,uw)))

prob = ODEProblem(mysys,u0,(0.0,6e5),[])

println("Beginning solving")
numRuns = 4
#sobolParams = generateLotsaLarter.makeSobolSeqParams(numRuns, batch+2)
sobolParams = [0.275 0.67 1.0 1.0 2.0 -0.7 6.7 0.53 0.36 2.0 0.25; 0.425 0.67 1.0 1.0 2.0 -0.7 6.7 0.53 0.36 2.0 0.25; 0.35 0.67 1.0 1.0 2.0 -0.7 6.7 0.48 0.36 2.0 0.25; 0.35 0.67 1.0 1.0 2.0 -0.7 6.7 0.54 0.36 2.0 0.25]
global goodRuns = 0

for run in 1:numRuns
	p_new = prob.p
	for t_index in 1:78
		p_new[(t_index-1)*11+1] = sobolParams[run, 1]
		p_new[(t_index-1)*11+2] = sobolParams[run, 2]
		p_new[(t_index-1)*11+3] = sobolParams[run, 3]
		p_new[(t_index-1)*11+4] = sobolParams[run, 4]
		p_new[(t_index-1)*11+5] = sobolParams[run, 5]
		p_new[(t_index-1)*11+6] = sobolParams[run, 6]
		p_new[(t_index-1)*11+7] = sobolParams[run, 7]
		p_new[(t_index-1)*11+8] = sobolParams[run, 8]
		p_new[(t_index-1)*11+9] = sobolParams[run, 9]
		p_new[(t_index-1)*11+10] = sobolParams[run, 10]
		p_new[(t_index-1)*11+11] = sobolParams[run, 11]
	end
	prob2 = remake(prob;p=p_new)
	sol2 = solve(prob2,AutoVern7(Rodas5()),saveat=1,maxiters=1e6)

	if generateLotsaLarter.checkForExpansion(sol2[1:3:end, :])
		println("Run "*string(run)*" failed because expanding")
	elseif generateLotsaLarter.checkForNoOscillation(sol2[1:3:end, :], 0.02)
		println("Run "*string(run)*" failed because flatlined")
	elseif length(sol2[1, :]) < 6e5
		println("Run "*string(run)*" failed because of maxiter problem")
	else
		bold = boldsignal(sol2.t/1000, sol2[1:3:end, :]) #switch to time series - turn off for now
		bold = generateLotsaLarter.filterTimeSeries(bold)
		cc_matr = cor(bold[:, 350:end], dims=2)
		if generateLotsaLarter.checkForUniformOscillation(cc_matr)
			println("Run "*string(run)*" failed because of uniform oscillations")
		else
			global goodRuns = goodRuns + 1
			file = matopen(savepath*"init1_run"*lpad(string(goodRuns), 4, '0')*".mat", "w")
			# save out LFP and params
			write(file, "cc", cc_matr)
			write(file, "params", sobolParams[run, :])
			close(file)
			println("Run "*string(run)*" completed!")
		end
	end

end


