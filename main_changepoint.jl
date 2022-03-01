cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random; using Base:@kwdef;
theme(:wong2)

include("VRPF.jl");include("CP.jl");include("BlockVRPF.jl")

# Generating the artificial data
J,y = CP.SimData(seed=1313)

T = collect(0:100:1000)
# Learn the λ and Σ
λ,Σ,_ = VRPF.TunePars(CP,y,T,θ0=[0.9,1.0,0.71,4.0,10.0],SMCAdaptN=10)

θ0 = rand.(CP.prior)
PG_VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=10)
PG_BlockVRPF_10Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[2.0,1.0],SMCN=10)

plot(PG_BlockVRPF_10Particles[:,4])

VRPF.log_pdmp_posterior(J,1000.0,y,CP,CP.pars())

plot(PG_VRPF_10Particles[:,5])