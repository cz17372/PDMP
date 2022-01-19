using Distributions, Plots, StatsPlots, Random, StatsBase
include("SNC.jl");include("VRPF.jl");include("ChangePoint.jl");include("BlockVRPF.jl")
J,y = ChangePoint.SimData(seed=2022)
N = 10000; T = collect(0:100.0:1000);auxpar=[2.0,1.0];par=ChangePoint.pars()

λ,Σ,_ = BlockVRPF.TunePars(ChangePoint,y,T,method="Global",auxpar=[2.0,1.0],Globalalpha=0.25,SMCAdaptN=100)
θ0 = rand.(ChangePoint.prior)
BlockR = BlockVRPF.PG(ChangePoint,y,T,proppar=(λ,Σ),auxpar=[2.0,1.0],θ0=θ0,SMCN=5,NFold=100)
R = VRPF.PG(ChangePoint,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=5,NFold=100)

plot(BlockR[:,1])