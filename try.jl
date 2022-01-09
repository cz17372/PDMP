model = ChangePoint
par = model.pars()
mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end
TimeVec = collect(0:50:1000)
N,T = size(SMCR.Weights)
BSWeight = zeros(N,T)
BSWeight[:,T] = SMCR.NWeights[:,T]
ParticleIndex = zeros(Int64,T)
ParticleIndex[T] = sample(1:N,Weights(BSWeight[:,T]),1)[1]
Laterξ = SMCR.Particles[ParticleIndex[T],T].X
L = Vector{Any}(undef,T)
L[T] = SMCR.Particles[ParticleIndex[T],T]
t = T-6
for i = 1:N
    BSWeight[i,t] = SMCR.Weights[i,t]+model.BlockBSIncrementalWeight(SMCR.PDMP[i,t],L[t+1],Laterξ,y,TimeVec[t],TimeVec[t+1],TimeVec[end],par)
end
BSWeight[:,t] = exp.(BSWeight[:,t] .- findmax(BSWeight[:,t])[1])
BSWeight[:,t] = BSWeight[:,t] / sum(BSWeight[:,t])
ParticleIndex[t] = sample(1:N,Weights(BSWeight[:,t]),1)[1]
L[t] = SMCR.Particles[ParticleIndex[t],t]
if L[t+1].M == 1
    Laterξ = PDMP(L[t].X.K + 1 + Laterξ.K,[L[t].X.τ;[L[t+1].taum];Laterξ.τ],[L[t].X.ϕ;[L[t+1].phim];Laterξ.ϕ])
else
    if isnothing(L[t+1].taum)
        Laterξ = PDMP(L[t].X.K + Laterξ.K,[L[t].X.τ;Laterξ.τ],[L[t].X.ϕ;Laterξ.ϕ])
    else
        Laterξ = PDMP(L[t].X.K + Laterξ.K,[L[t].X.τ[1:end-1];[L[t+1].taum];Laterξ.τ],[L[t].X.ϕ[1:end-1];[L[t+1].phim];Laterξ.ϕ])
    end
end
