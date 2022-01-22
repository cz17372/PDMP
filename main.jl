cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")
include("BlockVRPF.jl")

ξ,y = ChangePoint.SimData(seed=2022);ChangePoint.ProcObsGraph(ξ,y)
T = collect(0:50:1000)
θ = [0.9,1.0,0.75,4.0,10.0]


BVRPF_25par_100fold = BlockVRPF.PG(ChangePoint,y,T,auxpar=[2.0,1.0],SMCN=25,SMCAdaptN=100,Globalalpha=0.25)

plot(BVRPF_25par_100fold[:,4],color=:darkolivegreen,linewidth=0.2)

density(BVRPF_25par_100fold[:,5])

AC = autocor(BVRPF_25par_100fold)
plot(AC[:,2])

R = BlockVRPF.TunePars(ChangePoint,y,T,method="Global",auxpar=[2.0,1.0],SMCAdaptN=100)

@load "error.jld2"

a = ChangePoint.ProposedZDendity(Z[1,7],J[1,6],500.0,600.0,700.0,y,par,[2.0,1.0])
ChangePoint.BlockIncrementalWeight(J[1,5],Z[1,6],400,500,600,y,par,[2.0,1.0],a)