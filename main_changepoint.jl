cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,LaTeXStrings,Measures; using Base:@kwdef;
theme(:ggplot2)
include("VRPF.jl");include("CP.jl");include("BlockVRPF.jl")

J,y = CP.SimData(seed=1313)
scatter(y.t,y.y,markersize=2.0,makerstrokewidth=0.0,label="",color=:grey,framestyle=:box,xlabel="t",ylabel="PDMP")
extendedτ = [J.τ;[1000.0]]
for i = 1:(J.K+1)
    plot!(extendedτ[i:i+1],repeat(J.ϕ[i:i],2),label="",color=:red,linewidth=2.0)
end
current()

T = collect(0:100:1000)
θ0 = rand.(CP.prior)
λ,Σ,_ = VRPF.TunePars(CP,y,T,θ0=[0.9,1.0,0.71,4.0,10.0],SMCAdaptN=5,NAdapt=50000)

θ0 = rand.(CP.prior)
@save "proppar.jld2" λ Σ θ0
VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=10)
Random.seed!(17372)
BlockVRPF_100Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.05],SMCN=100,NFold=500)
Random.seed!(17370)
VRPF_100Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=100,NFold=500)
Info = "N=100,auxpar=[0.1,0.05],T=0:100:1000 VRPFseed=17370, BlockVRPFseed=17372"
@save "CPRes_100Particles1.jld2" Info BlockVRPF_100Particles VRPF_100Particles


Random.seed!(309)
BlockVRPF_50Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.05],SMCN=50,NFold=500)
Random.seed!(310)
VRPF_50Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=50,NFold=500)
Info = "N=50,auxpar=[0.1,0.05],T=0:100:1000 VRPFseed=309, BlockVRPFseed=310"
@save "CPRes_500Particles1.jld2" Info BlockVRPF_50Particles VRPF_50Particles

Random.seed!(220)
BlockVRPF_10Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.2],SMCN=25,NFold=500)
Random.seed!(221)
VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=25,NFold=500)
Info = "N=10,auxpar=[0.1,0.05],T=0:100:1000 VRPFseed=1019, BlockVRPFseed=1018"
@save "CPRes_10Particles1.jld2" Info BlockVRPF_10Particles VRPF_10Particles


density!(VRPF_10Particles[:,5])
density!(BlockVRPF_10Particles[:,5])

autocor(VRPF_10Particles)

autocor(BlockVRPF_10Particles)

plot(BlockVRPF_10Particles[:,4])
density(BlockVRPF_10Particles[:,1])

BlockVRPF_10Particles2 = BlockVRPF_10Particles

@load "CPRes_10Particles1.jld2"

p1 = plot(BlockVRPF_10Particles[:,5],label="par for backward kernel = [0.1,0.05]",size=(600,300))
p2 = plot(BlockVRPF_10Particles2[:,5],label="par for backward kernels = [0.1,0.2]",size=(600,300))
plot(p1,p2,layout=(2,1),size=(600,600))

AC1 = autocor(BlockVRPF_10Particles)
AC2 = autocor(BlockVRPF_10Particles2)

p1 = plot(AC1[:,1],label="back kernel par = [0.1,0.05]",size=(400,400))
plot!(AC2[:,1],label="back kernenl par = [0.1,0.2")
p2 = plot(AC1[:,2],label="back kernel par = [0.1,0.05]",size=(400,400))
plot!(AC2[:,2],label="back kernenl par = [0.1,0.2")
p3 = plot(AC1[:,3],label="back kernel par = [0.1,0.05]",size=(400,400))
plot!(AC2[:,3],label="back kernenl par = [0.1,0.2")
p4 = plot(AC1[:,4],label="back kernel par = [0.1,0.05]",size=(400,400))
plot!(AC2[:,4],label="back kernenl par = [0.1,0.2")
p5 = plot(AC1[:,5],label="back kernel par = [0.1,0.05]",size=(400,400))
plot!(AC2[:,5],label="back kernenl par = [0.1,0.2")

plot(p1,p2,p3,p4,p5,layout=(1,5),size=(2000,400))