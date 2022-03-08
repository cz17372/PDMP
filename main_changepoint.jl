cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random; using Base:@kwdef;
using Measures
theme(:wong2)
include("VRPF.jl");include("SNC.jl");include("BlockVRPF.jl")

J,y = SNC.SimData(seed=220)
T = collect(0:100:1000)
θ0 = rand.(SNC.prior)./[100,1,100]
λ,Σ = BlockVRPF.TunePars(SNC,y,T,θ0=[1/40,2/3,1/100],auxpar=[2.0,1.0],SMCAdaptN=10)
λ = λ[end]
θ0 = rand.(SNC.prior)./[200,1,200]

BlockVRPF_10Particles = BlockVRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,auxpar=[2.0,1.0],SMCN=10)
VRPF_10Particles = VRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=10)

plot(BlockVRPF_10Particles[:,3])


t = collect(0:0.1:1000)
ζ = SNC.getζ.(t,Ref(J),Ref(SNC.pars()))
p1 = plot(t,ζ,label="",color=:darkolivegreen,framestyle=:box,xlabel="t",ylabel="\\zeta",size=(1200,400),margin=15pt,linewidth=3.0)
p2 = histogram(y,bins=400,label="",xlabel="t",ylabel="Frequency",size=(1200,400),framestyle=:box)
plot(p1,p2,layout=(2,1),size=(1200,800))
savefig("SNC_Data.pdf")