cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,ForwardDiff; using Base:@kwdef;theme(:ggplot2)
include("SNC.jl");include("VRPF.jl");include("BlockVRPF.jl");include("ChangePoint.jl")
J,y = SNC.SimData(seed=220);T = collect(0.0:20:1000)
par = SNC.pars()
θ0 = rand.(SNC.prior)./[100,10,100]
R = VRPF.PG(SNC,y,T,θ0=θ0,SMCAdaptN=5,NBurn=20000,SMCN=5)

R = VRPF.SMC(2000,T,y,model=SNC,par=par)
BlockR = BlockVRPF.SMC(2000,T,y,model=SNC,par=par,auxpar=[0.01,0.01])
BS = BlockVRPF.BS(BlockR,y,T,model=SNC,par=par,auxpar=[0.01,0.01])
J2 = BS.BackwardPath
t = collect(0:0.1:1000.0)
ζ2 = SNC.getζ.(t,Ref(J),Ref(par))
plot(t,ζ2)
histogram(y,bins=400)
1/sum(BlockR.NWeights[:,1].^2)
1/sum(R.NWeights[:,1].^2)

R = BlockVRPF.TunePars(SNC,y,T,method="Global",θ0=θ0,auxpar=[0.01,0.01],SMCAdaptN=10)

@load "err.jld2"
f(posterior.lower)
d = truncated(Normal(),3.0,Inf)
g(x) = cdf(d,x)
g(3.0)
d = Gamma(12.0,0.06848371687502584)
plot(d)

@load "error.jld2"