cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")

ξ,y = ChangePoint.SimData(seed=124)
T = collect(0:100:1000)
ChangePoint.ProcObsGraph(ξ,y)

R = VRPF.PG(ChangePoint,y,T,method="Global",SMCN=5)

@save "../data/ChangePoint_data.jld2" ξ y