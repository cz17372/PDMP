using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")

ξ,y = ChangePoint.SimData(seed=124)
T = collect(0:100:1000)
ChangePoint.ProcObsGraph(ξ,y)


R = VRPF.PG(ChangePoint,y,collect(0:100.0:1000.0),method="Global",SMCN = 5)

plot(R[:,5])
density(R[:,5],label="",linewidth=2.0,color=:grey); vline!([10.0],label="",color=:red,linewidth=2.0)