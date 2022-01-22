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

Random.seed!(22532)
R = BlockVRPF.TunePars(ChangePoint,y,T,method="Global",auxpar=[2.0,1.0],SMCAdaptN=100)

@load "bserror.jld2"

_,index = findmax(SMCR.NWeights[:,end])
SMCR.Particles[:,end]

Z = SMCR.Particles[1,8]
J0 = SMCR.PDMP[1,7]
a = ChangePoint.ProposedZDendity(Z,J0,700,800,900,y,par,auxpar)
ChangePoint.BlockIncrementalWeight(J0,Z,700,800,900,y,par,auxpar,a)