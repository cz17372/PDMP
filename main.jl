cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")
include("BlockVRPF.jl")

ξ,y = ChangePoint.SimData(seed=2022);ChangePoint.ProcObsGraph(ξ,y)
T = collect(0:100:1000)

θ = [0.9,1.0,0.75,4.0,10.0]
BlockVRPF_RES = BlockVRPF.PG(ChangePoint,y,T,auxpar=[2.0,1.0],method="Component",SMCAdaptN=5,SMCN=5)
t = 3
p1 = plot(BlockVRPF_RES[:,t],size=(600,600),color=:grey,linewidth=0.5);hline!([θ[t]],linewidth=2.0,color=:red,label="")
p2 = density(BlockVRPF_RES[:,t],label="",color=:grey,linewidth=2.0);vline!([θ[t]],label="",color=:red,linewidth=2.0);
plot(p1,p2,layout=(1,2),size=(1200,600))

VRPF_RES = VRPF.PG(ChangePoint,y,T,SMCAdaptN=10,SMCN=5,Globalalpha=0.25)
VRPF.TunePars(ChangePoint,y,T,method="Global",Globalalpha=0.25,SMCAdaptN=100)

AC_BVRPF = autocor(BlockVRPF_RES)
AC_VRPF = autocor(VRPF_RES)

t = 1
p1 = plot(VRPF_RES[:,t],size=(600,600),color=:grey,linewidth=0.5);hline!([θ[t]],linewidth=2.0,color=:red,label="")
p2 = density(VRPF_RES[:,t],label="",color=:grey,linewidth=2.0);vline!([θ[t]],label="",color=:red,linewidth=2.0);
plot(p1,p2,layout=(1,2),size=(1200,600))

