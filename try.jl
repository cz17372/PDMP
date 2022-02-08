cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2, Optim
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint2.jl")
include("VRPF.jl")
include("BlockVRPF.jl")
ζ,y = CP.SimData(seed=1313);CP.ProcObsGraph(ζ,y)
T = collect(0.0:20:1000)
λ,Σ,_ = BlockVRPF.TunePars(CP,y,T,method="Global",auxpar=[2.0,1.0],SMCAdaptN=10,Globalalpha=0.2)
Random.seed!(12345)
θ0 = rand.(CP.prior)
BlockR_5Par = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),auxpar=[1.0,0.5],θ0=θ0,SMCN=5,NBurn=20000)
R_5Par = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=5,NBurn=20000)
BlockR_25Par = BlockVRPF.PG(CP,y,T,proppar =(λ,Σ),auxpar=[0.01,0.25],θ0=θ0,SMCN=25,NBurn=20000)
R_25Par = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=25,NBurn=20000)
autocor(BlockR_5Par)
autocor(R_5Par)

plot(BlockR_5Par[:,4],color=:grey,linewidth=0.5)

density(BlockR_5Par[:,5])