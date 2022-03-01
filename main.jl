cd(dirname(@__FILE__))
using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random; using Base:@kwdef;
theme(:wong2)

include("VRPF.jl");include("SNC.jl");include("BlockVRPF.jl")
t = collect(0:0.1:1000);par = SNC.pars();T=collect(0:100:1000)
J,y = SNC.SimData(seed=2029);ζ0 = SNC.getζ.(t,Ref(J),Ref(par))
plot(t,ζ0,framestyle=:box,label="",size=(1500,800),linewidth=2.0);vline!(collect(0:100:1000),color=:grey,label="",linestyle=:dash)

X,llk = VRPF.KX(100.0,y,SNC,par)


N = 20;M = 100
R = VRPF.SMC(N,T,y,SNC,par)
BR = BlockVRPF.SMC(N,T,y,SNC,par,[1.0,0.1])
VRPF_proc = zeros(M,length(t))
BVRPF_proc = zeros(M,length(t))

for i = 1:M
    println(i)
    J = VRPF.BS(R,y,T,SNC,par).Path
    ζ = SNC.getζ.(t,Ref(J),Ref(par))
    VRPF_proc[i,:] = ζ
    J = BlockVRPF.BS(BR,y,T,SNC,par,[1.0,0.1]).Path
    ζ = SNC.getζ.(t,Ref(J),Ref(par))
    BVRPF_proc[i,:] = ζ
end

plot(t,ζ0,framestyle=:box,label="",size=(1500,800),linewidth=5.0);vline!(collect(0:100:1000),color=:grey,label="",linestyle=:dash)
for i = 1:M
    plot!(t,VRPF_proc[i,:],color=:red,label="",linewidth=0.5)
    plot!(t,BVRPF_proc[i,:],color=:green,label="",linewidth=0.5)
end
current()



function AncestralIndices(R)
    N,P = size(R.W)
    output = zeros(Int64,N,P)
    for i = 1:N
        output[i,P] = i
        for j = (P-1):-1:1
            output[i,j] = R.A[output[i,j+1],j]
        end
    end
    plot(output[1,:],label="",linewidth=0.5,color=:grey)
    for i = 2:N
        plot!(output[i,:],label="",linewidth=0.5,color=:grey)
    end
    current()
end
            
AncestralIndices(R)

N = 10
X = Array{Any,1}(undef,N)
llk = zeros(N)
W = zeros(N)
J = Array{Any,1}(undef,N)
for i = 1:N
    X[i],llk[i] = VRPF.KX(100.0,y,SNC,par)
    J[i] = X[i]
    W[i] = VRPF.G(X[i],100.0,y,SNC,par,llk[i])
end
W = exp.(W) / sum(exp.(W))
A = vcat(fill.(1:N,rand(Multinomial(N,W)))...)
VRPFX2 = Array{Any,1}(undef,N)
BVRPFX2 = Array{Any,1}(undef,N)
VRPFJ2  = Array{Any,1}(undef,N)
BVRPFJ2 = Array{Any,1}(undef,N)
for i = 1:N
    VRPFX2[i],_ = VRPF.KX(100,200,J[A[i]],y,SNC,par)
    VRPFJ2[i] = VRPF.addPDMP(J[A[i]],VRPFX2[i])
    BVRPFX2[i],_ = BlockVRPF.KZ(0.0,100.0,200.0,J[A[i]],y,SNC,par,[2.0,1.0])
    BVRPFJ2[i],_ = BlockVRPF.AddPDMP(J[A[i]],BVRPFX2[i])
end

t = collect(0:0.1:200)
plot(t,ζ0[1:2001])
VRPF_proc = zeros(N,length(t))
BVRPF_proc = zeros(N,length(t))
for i = 1:N
    VRPF_proc[i,:] = SNC.getζ.(t,Ref(VRPFJ2[i]),Ref(par))
    BVRPF_proc[i,:] = SNC.getζ.(t,Ref(BVRPFJ2[i]),Ref(par))
end
plot(t,ζ0[1:2001],label="",linewidth=3.0);vline!([100],linestyle=:dash,label="",xlabel="Time",ylabel="PDMP",framestyle=:box)
for i = 1:N
    if i == 1
        plot!(t,VRPF_proc[i,:],color=:red,label="VRPF",linewidth=0.5)
        plot!(t,BVRPF_proc[i,:],color=:green,label="BlockVRPF",linewidth=0.5)
    else
        plot!(t,VRPF_proc[i,:],color=:red,label="",linewidth=0.5)
        plot!(t,BVRPF_proc[i,:],color=:green,label="",linewidth=0.5)
    end
end
current()
savefig("BlockVRPF_Illustration.pdf")