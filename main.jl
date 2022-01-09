cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
using Base:@kwdef
theme(:ggplot2)
include("ChangePoint.jl")
include("VRPF.jl")
include("BlockVRPF.jl")

ξ,y = ChangePoint.SimData(seed=1313);#ChangePoint.ProcObsGraph(ξ,y)
VRPF.PG(ChangePoint,y,collect(0:100:1000),method="Global",NFold=50)
T = collect(0:100:1000)
SMCR = BlockVPRF.SMC(1000,T,y,model=ChangePoint,par=ChangePoint.pars());
BR = BlockVPRF.BS(SMCR,y,T,model=ChangePoint,par=ChangePoint.pars())

p=ChangePoint.ProcObsGraph(BR.BackwardPath,y);ChangePoint.PlotPDMP(p,ξ,1000.0)
Index = zeros(1000,10)
for n = 1:1000
    println(n)
    Index[n,:] = BlockVPRF.BS(SMCR,y,T,model=ChangePoint,par=ChangePoint.pars()).BackIndex
end
plot(Index[1,:],label="",color=:grey)
for n= 2:1000
    plot!(Index[n,:],label="",color=:grey)
end

R = BlockVPRF.PG(ChangePoint,y,T,method="Global",SMCN=5) # (Time: 00:38:26)
R2 = VRPF.PG(ChangePoint,y,T,method="Global",SMCN=5)
density(R[:,1])

plot(R[:,5])

RES = autocor(R)
RES2 = autocor(R2)
t  = 1 ; plot(RES[:,t]);plot!(RES2[:,t])

plot(RES[:,5])