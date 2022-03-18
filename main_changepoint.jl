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
@load "proppar.jld2"
Random.seed!(17372)
BlockVRPF_100Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.05],SMCN=100,NFold=500)
Random.seed!(17370)
VRPF_100Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=100,NFold=500)
Info = "N=100,auxpar=[0.1,0.05],T=0:100:1000 VRPFseed=17370, BlockVRPFseed=17372"
@save "CPRes_100Particles1.jld2" Info BlockVRPF_100Particles VRPF_100Particles
@load "CPRes_100Particles1.jld2"
autocor(BlockVRPF_100Particles)
autocor(BlockVRPF_100Particles2)
autocor(VRPF_100Particles)
Random.seed!(309)
BlockVRPF_50Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.05],SMCN=50,NFold=500)
Random.seed!(310)
VRPF_50Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=50,NFold=500)
Info = "N=50,auxpar=[0.1,0.05],T=0:100:1000 VRPFseed=309, BlockVRPFseed=310"
@save "CPRes_500Particles1.jld2" Info BlockVRPF_50Particles VRPF_50Particles

Random.seed!(220)
BlockVRPF_10Particles = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.05],SMCN=25,NFold=500)
Random.seed!(221)
VRPF_10Particles = VRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,SMCN=25,NFold=500)
Info = "N=10,auxpar=[0.1,0.05],T=0:100:1000 VRPFseed=1019, BlockVRPFseed=1018"
@save "CPRes_10Particles1.jld2" Info BlockVRPF_10Particles VRPF_10Particles

Random.seed!(220)
T = collect(0:100:1000)
BlockVRPF_10Particles2 = BlockVRPF.PG(CP,y,T,proppar=(λ,Σ),θ0=θ0,auxpar=[0.1,0.2],SMCN=25,NFold=500)
density!(VRPF_10Particles[:,5])
density!(BlockVRPF_10Particles[:,5])

autocor(VRPF_10Particles)

autocor(BlockVRPF_10Particles)

plot(BlockVRPF_10Particles[:,4])
density(BlockVRPF_10Particles[:,1])

BlockVRPF_10Particles2 = BlockVRPF_10Particles

@load "CPRes_10Particles1.jld2"
@load "CPRes_500Particles1.jld2"
@load "CPRes_100Particles1.jld2"

p1 = plot(BlockVRPF_10Particles[:,5],label="par for backward kernel = [0.1,0.05]",size=(600,300))
p2 = plot(BlockVRPF_10Particles2[:,5],label="par for backward kernels = [0.1,0.2]",size=(600,300))
plot(p1,p2,layout=(2,1),size=(600,600))

AC1 = autocor(BlockVRPF_10Particles)
AC2 = autocor(BlockVRPF_10Particles2)

p1 = plot(AC1[:,1],size=(400,400),xlabel=L"Lag $(\rho)$",ylabel="Autocorrelations",xguidefontsize=20,yguidefontsize=20,color=:grey,linewidth=3.0,label="")
plot!(AC2[:,1],color=:darkolivegreen,linewidth=3.0,label="")
p2 = plot(AC1[:,2],color=:grey,linewidth=3.0,size=(400,400),xlabel=L"Lag $(\sigma_\phi)$",xguidefontsize=20,yguidefontsize=20,label="")
plot!(AC2[:,2],color=:darkolivegreen,linewidth=3.0,label="")
p3 = plot(AC1[:,3],color=:grey,linewidth=3.0,size=(400,400),xlabel=L"Lag $(\sigma_y)$",xguidefontsize=20,yguidefontsize=20,label="")
plot!(AC2[:,3],color=:darkolivegreen,linewidth=3.0,label="")
p4 = plot(AC1[:,4],color=:grey,linewidth=3.0,size=(400,400),xlabel=L"Lag $(\alpha)$",ylabel="Autocorrelations",xguidefontsize=20,yguidefontsize=20,label="")
plot!(AC2[:,4],color=:darkolivegreen,linewidth=3.0,label="")
p5 = plot(AC1[:,5],color=:grey,linewidth=3.0,size=(400,400),xlabel=L"Lag $(\beta)$",xguidefontsize=20,yguidefontsize=20,label="")
plot!(AC2[:,5],color=:darkolivegreen,linewidth=3.0,label="")

plot(p1,p2,p3,p4,p5,layout=l,size=(1200,800),margins=20pt)
savefig("../PDMP-writeup/PG-simulation-CPRes2.pdf")
t = 1
p1 = density(BlockVRPF_10Particles[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"$\rho$",ylabel="Density",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
density!(BlockVRPF_100Particles[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
density!(VRPF_100Particles[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
density!(VRPF_50Particles[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
density!(VRPF_10Particles[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)
vline!([θstar[t]],label="",color=:grey,linewidth=2.0,linestyle=:dash)
t = 2
p2 = density(BlockVRPF_10Particles[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"$\sigma_\phi$",ylabel="",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
density!(BlockVRPF_100Particles[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
density!(VRPF_100Particles[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
density!(VRPF_50Particles[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
density!(VRPF_10Particles[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)
vline!([θstar[t]],label="",color=:grey,linewidth=2.0,linestyle=:dash)
t = 3
p3 = density(BlockVRPF_10Particles[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"$\sigma_y$",ylabel="",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
density!(BlockVRPF_100Particles[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
density!(VRPF_100Particles[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
density!(VRPF_50Particles[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
density!(VRPF_10Particles[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)
vline!([θstar[t]],label="",color=:grey,linewidth=2.0,linestyle=:dash)

t = 4
p4 = density(BlockVRPF_10Particles[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"$\alpha$",ylabel="Density",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
density!(BlockVRPF_100Particles[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
density!(VRPF_100Particles[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
density!(VRPF_50Particles[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
density!(VRPF_10Particles[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)
vline!([θstar[t]],label="",color=:grey,linewidth=2.0,linestyle=:dash)

t = 5
p5 = density(BlockVRPF_10Particles[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"$\beta$",ylabel="",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
density!(BlockVRPF_100Particles[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
density!(VRPF_100Particles[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
density!(VRPF_50Particles[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
density!(VRPF_10Particles[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)
vline!([θstar[t]],label="",color=:grey,linewidth=2.0,linestyle=:dash)
l = @layout [a b c; d e _]
plot(p1,p2,p3,p4,p5,layout=l,size=(1500,1000),margin=30pt)
savefig("../PDMP-writeup/PG-simulation-CPRes1.pdf")
BAC1 = autocor(BlockVRPF_10Particles)
BAC2 = autocor(BlockVRPF_50Particles)
BAC3 = autocor(BlockVRPF_100Particles)
AC1 = autocor(VRPF_10Particles)
AC2 = autocor(VRPF_50Particles)
AC3 = autocor(VRPF_100Particles)
t = 1 
p1 = plot(BAC1[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"Lag $(\rho)$",ylabel="Autocorrelations",xguidefontsize=20,yguidefontsize=20)
plot!(BAC2[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
plot!(BAC3[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
plot!(AC3[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
plot!(AC2[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
plot!(AC1[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)
t = 2
p1 = plot(BAC1[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"Lag $(\sigma_\phi)$",ylabel="",xguidefontsize=20,yguidefontsize=20)
plot!(BAC2[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
plot!(BAC3[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
plot!(AC3[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
plot!(AC2[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
plot!(AC1[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)


t = 3
p1 = plot(BAC1[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"Lag $(\sigma_y)$",ylabel="",xguidefontsize=20,yguidefontsize=20)
plot!(BAC2[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
plot!(BAC3[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
plot!(AC3[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
plot!(AC2[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
plot!(AC1[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)

t = 4
p1 = plot(BAC1[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"Lag $(\alpha)$",ylabel="Autocorrelations",xguidefontsize=20,yguidefontsize=20)
plot!(BAC2[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
plot!(BAC3[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
plot!(AC3[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
plot!(AC2[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
plot!(AC1[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)

t = 5
p1 = plot(BAC1[:,t],label="",linestyle=:dot,color=:darkolivegreen,linewidth=2.0,size=(500,500),xlabel=L"Lag $(\beta)$",ylabel="",xguidefontsize=20,yguidefontsize=20)
plot!(BAC2[:,t],label="",linestyle=:dash,color=:darkolivegreen,linewidth=2.0)
plot!(BAC3[:,t],label="",linestyle=:solid,color=:darkolivegreen,linewidth=2.0)
plot!(AC3[:,t],label="",linestyle=:solid,color=:red,linewidth=2.0)
plot!(AC2[:,t],label="",linestyle=:dash,color=:red,linewidth=2.0)
plot!(AC1[:,t],label="",linestyle=:dot,color=:red,linewidth=2.0)