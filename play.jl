cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,LaTeXStrings,Measures; using Base:@kwdef;
theme(:ggplot2)
include("VRPF.jl");include("SNC.jl");include("BlockVRPF.jl")
par = SNC.pars()
J,y = SNC.SimData(seed=19931018)
t = collect(0:0.1:250)
ζ = SNC.getζ.(t,Ref(J),Ref(par))
plot(t,ζ)
T = [0.0,30.0,50,105,150,210,250]
vline!(T,color=:red,label="")

R = VRPF.SMC(100,T,y,SNC,par)
BR = BlockVRPF.SMC(100,T,y,SNC,par,[2.0,1.0])
BSPath = VRPF.BS(R,y,T,SNC,par).Path
BSRPath = BlockVRPF.BS(BR,y,T,SNC,par,[2.0,1.0]).Path
ζR = SNC.getζ.(t,Ref(BSPath),Ref(par))
ζBR = SNC.getζ.(t,Ref(BSRPath),Ref(par))
plot(t,ζ,color=:grey,label="")
plot!(t,ζR,color=:red,label="")
plot!(t,ζBR,color=:green,label="")
