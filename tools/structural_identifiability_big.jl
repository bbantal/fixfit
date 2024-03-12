using StructuralIdentifiability

# Same equations as used in the paper with a couple important modifications:
# 1. u(t) is modeled as a constant because it StructuralIdentifiability doesn't support dirac deltas, but this is close enough.
# 2. β(t) is simplified because StructuralIdentifiability doesn't support raising states to powers at the moment. The changes are very small though, so it doesn't impact much.
# 3. If you want to try observing both glucose and insulin levels uncomment the second observer function below.
ode = @ODEmodel(
    x1'(t) = 5.4*(C+Sᵢ*(p*322.0*(5.4^2/(α^2+5.4^2))/γ)) - (C+Sᵢ*x2(t))*x1(t) + 0.03 + u,
    x2'(t) = p*x3(t)*(1/(1+(α/x1(t))^2))-γ*x2(t),
    x3'(t) = x3(t)*(0.0000001*x1(t)),
    y1(t) = x1(t)#,
    #y2(t) = x2(t)
)

println(assess_identifiability(ode))
println(assess_identifiability(ode, funcs_to_check = [Sᵢ*p])) # Check for known redundancy explicitly

# Show the reparameterized equations in minimal parameter space
reparam = reparametrize_global(ode)
@assert isempty(reparam[:implicit_relations])
reparam[:new_vars]

