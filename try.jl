
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

J0 = J[1,6];Z = Z[1,7]; t0=500;t1=600;t2=700
auxpar=[2.0,1.0]
llk = 0.0
MeanJumpTime = (t2-t1)/(par.α*par.β)
llk += logpdf(Poisson(MeanJumpTime),Z.X.K)
llk += sum(log.(collect(1:Z.X.K)/(t2-t1)))
IJ = Gamma(par.α,par.β)
prob = cdf(IJ,t1-J0.τ[end])