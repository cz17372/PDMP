using Distributions, Plots, StatsPlots,Random, StatsBase
include("ChangePoint.jl")
using .ChangePoint
include("VRPF.jl") 
ξ,y = ChangePoint.SimData(seed=4013)
@time R = VRPF.SMC(100,collect(0.0:100.0:1000.0),y,model=ChangePoint,par=pars());

X,l = ChangePoint.GenParticle(0.0,100.0,y,pars())
X2,l2 = ChangePoint.GenParticle(100.0,200.0,X,y,pars())

ChangePoint.CalSampDen(X2,100.0,200.0,X,y,pars())