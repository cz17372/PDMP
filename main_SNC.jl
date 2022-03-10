cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,LaTeXStrings,Measures; using Base:@kwdef;
theme(:ggplot2)
include("VRPF.jl");include("SNC.jl");include("BlockVRPF.jl")

J,y = SNC.SimData(seed=220)
T = collect(0:100:1000)
@load "proppar.jld2"
BlockVRPF_10Particles = BlockVRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,auxpar=[2.0,1.0],SMCN=10)
VRPF_10Particles = VRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=10)
Info = "T=0:100:1000,N=10,auxpar=[2.0,1.0]"
@save "SNC_10Particles1.jld2" BlockVRPF_10Particles VRPF_10Particles Info
BlockVRPF_50Particles = BlockVRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,auxpar=[2.0,1.0],SMCN=50)
VRPF_50Particles = VRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=50)
Info = "T=0:100:1000,N=50,auxpar=[2.0,1.0]"
@save "SNC_50Particles1.jld2" BlockVRPF_50Particles VRPF_50Particles Info
BlockVRPF_100Particles = BlockVRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,auxpar=[2.0,1.0],SMCN=100)
VRPF_100Particles = VRPF.PG(SNC,y,T,proppar=(λ,Σ),θ0 = θ0,SMCN=100)
Info = "T=0:100:1000,N=100,auxpar=[2.0,1.0]"
@save "SNC_100Particles1.jld2" BlockVRPF_100Particles VRPF_100Particles Info


@load "SNC_10Particles1.jld2"
@load "SNC_50Particles1.jld2"
@load "SNC_100Particles1.jld2"

t = 1
p1 = density(BlockVRPF_10Particles[:,t],linestyle=:dot,color=:darkolivegreen,linewidth=2.0,label="",framestyle=:box,size=(600,600),ylabel="Density",xlabel=L"$\lambda_{\tau}$",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],linestyle=:dash,color=:darkolivegreen,linewidth=2.0,label="")
density!(BlockVRPF_100Particles[:,t],linestyle=:solid,color=:darkolivegreen,linewidth=2.0,label="")
density!(VRPF_10Particles[:,t],linestyle=:dot,color=:red,linewidth=2.0,label="")
density!(VRPF_50Particles[:,t],linestyle=:dash,color=:red,linewidth=2.0,label="")
density!(VRPF_100Particles[:,t],linestyle=:solid,color=:red,linewidth=2.0,label="")
vline!([1/40],label="",color=:grey,linestyle=:dash,linewidth=2.0)

t = 2
p2 = density(BlockVRPF_10Particles[:,t],linestyle=:dot,color=:darkolivegreen,linewidth=2.0,label="",framestyle=:box,size=(600,600),xlabel=L"$\lambda_{\phi}$",ylabel="",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],linestyle=:dash,color=:darkolivegreen,linewidth=2.0,label="")
density!(BlockVRPF_100Particles[:,t],linestyle=:solid,color=:darkolivegreen,linewidth=2.0,label="")
density!(VRPF_10Particles[:,t],linestyle=:dot,color=:red,linewidth=2.0,label="")
density!(VRPF_50Particles[:,t],linestyle=:dash,color=:red,linewidth=2.0,label="")
density!(VRPF_100Particles[:,t],linestyle=:solid,color=:red,linewidth=2.0,label="")
vline!([2/3],label="",color=:grey,linestyle=:dash,linewidth=2.0)

t = 3
p3 = density(BlockVRPF_10Particles[:,t],linestyle=:dot,color=:darkolivegreen,linewidth=2.0,label="",framestyle=:box,size=(600,600),xlabel=L"$\kappa$",ylabel="",xguidefontsize=20,yguidefontsize=20)
density!(BlockVRPF_50Particles[:,t],linestyle=:dash,color=:darkolivegreen,linewidth=2.0,label="")
density!(BlockVRPF_100Particles[:,t],linestyle=:solid,color=:darkolivegreen,linewidth=2.0,label="")
density!(VRPF_10Particles[:,t],linestyle=:dot,color=:red,linewidth=2.0,label="")
density!(VRPF_50Particles[:,t],linestyle=:dash,color=:red,linewidth=2.0,label="")
density!(VRPF_100Particles[:,t],linestyle=:solid,color=:red,linewidth=2.0,label="")
vline!([1/100],label="",color=:grey,linestyle=:dash,linewidth=2.0)

plot(p1,p2,p3,layout=(1,3),size=(1800,600),margin=30pt)

savefig("SNC_Sim_posterior.pdf")

p1 = plot(VRPF_10Particles[:,1],label="",linewidth=0.5,color=:grey,xticks=:None)
p2 = plot(VRPF_50Particles[:,1],label="",linewidth=0.5,color=:grey)
p3 = plot(VRPF_100Particles[:,1],label="",linewidth=0.5,color=:grey)

BAC10 = autocor(BlockVRPF_10Particles)
BAC50 = autocor(BlockVRPF_50Particles)
BAC100= autocor(BlockVRPF_100Particles)
AC10  = autocor(VRPF_10Particles)
AC50  = autocor(VRPF_50Particles)
AC100 = autocor(VRPF_100Particles)

t=1
p1 = plot(BAC10[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:dot,size=(600,600),xlabel=L"Lag $(\lambda_\tau)$",ylabel="autocorrelation",xguidefontsize=20,yguidefontsize=20)
plot!(BAC50[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:dash)
plot!(BAC100[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:solid)
plot!(AC10[:,t],label="",color=:red,linewidth=2,linestyle=:dot)
plot!(AC50[:,t],label="",color=:red,linewidth=2,linestyle=:dash)
plot!(AC100[:,t],label="",color=:red,linewidth=2,linestyle=:solid)

t=2
p2 = plot(BAC10[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:dot,size=(600,600),xlabel=L"Lag $(\lambda_\phi)$",ylabel="",xguidefontsize=20,yguidefontsize=20)
plot!(BAC50[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:dash)
plot!(BAC100[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:solid)
plot!(AC10[:,t],label="",color=:red,linewidth=2,linestyle=:dot)
plot!(AC50[:,t],label="",color=:red,linewidth=2,linestyle=:dash)
plot!(AC100[:,t],label="",color=:red,linewidth=2,linestyle=:solid)

t=3
p3 = plot(BAC10[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:dot,size=(600,600),xlabel=L"Lag $(\kappa)$",ylabel="",xguidefontsize=20,yguidefontsize=20)
plot!(BAC50[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:dash)
plot!(BAC100[:,t],label="",color=:darkolivegreen,linewidth=2,linestyle=:solid)
plot!(AC10[:,t],label="",color=:red,linewidth=2,linestyle=:dot)
plot!(AC50[:,t],label="",color=:red,linewidth=2,linestyle=:dash)
plot!(AC100[:,t],label="",color=:red,linewidth=2,linestyle=:solid)

plot(p1,p2,p3,layout=(1,3),size=(1800,600),margin=30pt)
savefig("SNC_ACF.pdf")