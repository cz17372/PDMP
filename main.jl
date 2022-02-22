cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,ForwardDiff; using Base:@kwdef;
theme(:mute)

include("VRPF.jl");include("SNC.jl");include("BlockVRPF.jl")
t = collect(0:0.1:1000);par = SNC.pars()
J,y = SNC.SimData(seed=2022)
ζ = SNC.getζ.(t,Ref(J),Ref(par))
plot(t,ζ)