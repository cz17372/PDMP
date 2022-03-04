cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random; using Base:@kwdef;
theme(:wong2)

include("VRPF.jl");include("CP.jl");include("BlockVRPF.jl")

# Generating the artificial data
J,y = CP.SimData(seed=1313)

T = collect(0:100:1000)
# Learn the λ and Σ
λ2,Σ2,_ = BlockVRPF.TunePars(CP,y,T,θ0=[0.9,1.0,0.71,4.0,10.0],auxpar=[2.0,1.0],SMCAdaptN=10)

θ0 = rand.(CP.prior)
PG_VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=10)
PG_BlockVRPF_10Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[2.0,1.0],SMCN=10)

plot(PG_BlockVRPF_10Particles[:,4])

VRPF.log_pdmp_posterior(J,1000.0,y,CP,CP.pars())

plot(PG_VRPF_10Particles[:,5])
plot(PG_BlockVRPF_10Particles[:,5])
using StatsBase
autocor(PG_BlockVRPF_10Particles)
autocor(PG_VRPF_10Particles)
using JLD2
Info = "Seed=1313, N=10, auxpar=[2.0,1.0]"
@save "10particleRes.jld2" PG_BlockVRPF_10Particles PG_VRPF_10Particles Info

PG_VRPF_25Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=25)
PG_BlockVRPF_10Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[2.0,1.0],SMCN=25)
PG_BlockVRPF_25Particles = PG_BlockVRPF_10Particles
Info = "Seed=1313, N=25, auxpar=[2.0,1.0]"
@save "25particleRes.jld2" PG_BlockVRPF_25Particles PG_VRPF_25Particles Info
plot(PG_VRPF_25Particles[:,4])
plot(PG_BlockVRPF_25Particles[:,4])
autocor(PG_BlockVRPF_10Particles)
autocor(PG_VRPF_25Particles)

density(PG_BlockVRPF_25Particles[:,1]);density!(PG_VRPF_25Particles[:,1])

PG_VRPF_50Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=50)
PG_BlockVRPF_50Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[2.0,1.0],SMCN=50)
Info = "Seed=1313, N=50, auxpar=[2.0,1.0]"
@save "50particleRes.jld2" PG_BlockVRPF_50Particles PG_VRPF_50Particles Info

autocor(PG_VRPF_50Particles)

plot(PG_BlockVRPF_50Particles[:,5])

density(PG_VRPF_50Particles[:,2]);density!(PG_VRPF_25Particles[:,2]);density!(PG_VRPF_10Particles[:,2]);density!(PG_BlockVRPF_10Particles[:,2]);density!(PG_BlockVRPF_25Particles[:,2]);density!(PG_BlockVRPF_50Particles[:,2])


T = collect(0:10:1000)
PG_VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=10)
PG_BlockVRPF_10Particles = BlockVRPF.PG(CP,y,collect(0:100:1000),proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.1],SMCN=20)
Info = "Seed=1313, N=10, auxpar=[2.0,1.0],T=0:10:1000"
@save "10particleRes2.jld2" PG_BlockVRPF_10Particles PG_VRPF_10Particles Info

plot(PG_VRPF_10Particles[:,1])
density(PG_VRPF_10Particles[:,5])