
cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")
include("BlockVRPF.jl")
@load "error.jld2"
ξ,y = ChangePoint.SimData(seed=2022);ChangePoint.ProcObsGraph(ξ,y)
a = ChangePoint.ProposedZDendity(Z[1,7],J[1,6],500.0,600.0,700.0,y,par,[2.0,1.0])

J0 = J[1,9];Z = Z[1,10]; t0=800;t1=900;t2=1000

ChangePoint.ProposedZDendity(Z,J0,t0,t1,t2,y,par,[2.0,1.0])
ChangePoint.BlockIncrementalWeight(J0,Z,t0,t1,t2,y,par,[2.0,1.0],0.0)

R = BlockVRPF.SMC(100,collect(0:100:1000),y,model=ChangePoint,par=par,auxpar=[2.0,1.0])

_,index = findmax(R.NWeights[:,end])
R.PDMP[index,end]

BS = BlockVRPF.BS(R,y,collect(0:100:1000),model=ChangePoint,par=par,auxpar=[2.0,1.0])
BS.BackwardPath