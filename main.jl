cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")
include("BlockVRPF.jl")

ξ,y = ChangePoint.SimData(seed=1313);ChangePoint.ProcObsGraph(ξ,y)
VRPF.PG(ChangePoint,y,collect(0:100:1000),method="Global",NFold=50)
