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