cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")

ξ,y = ChangePoint.SimData(seed=124)
T = collect(0:100:1000)
ChangePoint.ProcObsGraph(ξ,y)