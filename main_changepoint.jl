cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,LaTeXStrings,Measures; using Base:@kwdef;
theme(:ggplot2)
include("VRPF.jl");include("CP.jl");include("BlockVRPF.jl")

J,y = CP.SimData(seed=12345)
scatter(y.t,y.y,markersize=2.0,makerstrokewidth=0.0,label="",color=:grey,framestyle=:box,xlabel="t",ylabel="PDMP")
extendedτ = [J.τ;[1000.0]]
for i = 1:(J.K+1)
    plot!(extendedτ[i:i+1],repeat(J.ϕ[i:i],2),label="",color=:red,linewidth=2.0)
end
current()

T = collect(0:10:1000)
θ0 = rand.(CP.prior)
λ,Σ,_ = VRPF.TunePars(CP,y,T,θ0=θ0,SMCAdaptN=5,NAdapt=50000)

θ0 = rand.(CP.prior)
@save "proppar.jld2" λ Σ θ0
VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=10)
BlockVRPF_10Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[2.0,1.0],SMCN=10)


density(VRPF_10Particles[:,3])