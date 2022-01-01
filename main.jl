using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra
using Base:@kwdef
include("ChangePoint.jl")
include("VRPF.jl")

ξ,y = ChangePoint.SimData(seed=17369)
T = collect(0:100:1000)
VRPF.TunePars(ChangePoint,y,T;SMCN=100,NAdapt=10000)