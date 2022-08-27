using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2,Random,LaTeXStrings,Measures 
using Base:@kwdef;
using ProgressMeter
theme(:wong2)


include("VRPF.jl");include("CP.jl");include("BlockVRPF.jl")
par = CP.pars();
T = [0.0,97.0,195.0,350,465,500];
J,y = CP.SimData(seed=1313)
p0=scatter(y.t[1:500],y.y[1:500],markersize=2.0,makerstrokewidth=0.0,label="",color=:grey,framestyle=:box,xlabel="t",ylabel="PDMP",xticks=T,size=(1000,500),margin=12.5pt)
extendedτ = [J.τ[1:15];[500.0]]
for i = 1:15
    plot!(extendedτ[i:i+1],repeat(J.ϕ[i:i],2),label="",color=:red,linewidth=2.0)
end
vline!(T,label="",color=:grey,linestyle=:dash)
current()


N = 200
VRPFζ = Array{Any}(undef,1000)
BlockVRPFζ = Array{Any}(undef,1000)
Random.seed!(1313)
@showprogress 1 for i = 1:1000
    R = VRPF.SMC(N,T,y,CP,par);
    VRPFζ[i] = VRPF.BS(R,y,T,CP,par).Path
end
Random.seed!(12345)
@showprogress 1 for i = 1:1000
    R = BlockVRPF.SMC(N,T,y,CP,par,[0.1,0.5]);
    BlockVRPFζ[i] = BlockVRPF.BS(R,y,T,CP,par,[0.1,0.5]).Path
end
P = length(T)-1
VRPFK = zeros(Int64,1000,P);
BlockVRPFK = zeros(Int64,1000,P);
for i = 1:1000
    for j = 1:P
        VRPFK[i,j] = length(findall(T[j] .< VRPFζ[i].τ .<= T[j+1]))
        BlockVRPFK[i,j] = length(findall(T[j] .< BlockVRPFζ[i].τ .<= T[j+1]))
    end
end
ToTVRPFK = zeros(Int64,1000)
ToTBlockVRPFK = zeros(Int64,1000)
for i = 1:1000
    ToTVRPFK[i] = VRPFζ[i].K
    ToTBlockVRPFK[i] = BlockVRPFζ[i].K 
end
p1 = histogram(ToTVRPFK,ylim=(0,250),color=:orange,bar_width=1,label="",bins=15,xlabel="K",ylabel="Frequency");vline!([14],label="",color=:red,linewidth=2.0,linestyle=:dash);
p2 = histogram(ToTBlockVRPFK,ylim=(0,250),color=:darkolivegreen,bar_width=1,label="",bins=15,xlabel="K");vline!([14],label="",color=:red,linewidth=2.0,linestyle=:dash);
plot(p1,p2,layout=(1,2))
EstimatedWholePDMP(14,VRPFζ,BlockVRPFζ,T,J)



function EstimatedPDMP(Block::Int64,NJumps::Int64,VRPFζ,BlockVRPFζ,T,J)
    VRPFτ = zeros(NJumps)
    VRPFϕ = zeros(NJumps)
    BlockVRPFτ = zeros(NJumps)
    BlockVRPFϕ = zeros(NJumps)
    VRPFn = 0
    BlockVRPFn = 0
    VRPFprevphi = 0.0
    BlockVRPFprevphi = 0.0
    M = length(VRPFζ)
    for i = 1:M
        VRPF_index = findall(T[Block] .< VRPFζ[i].τ .<= T[Block+1])
        if length(VRPF_index) == NJumps
            VRPFτ = VRPFτ .+ VRPFζ[i].τ[VRPF_index]
            VRPFϕ = VRPFϕ .+ VRPFζ[i].ϕ[VRPF_index]
            VRPFprevphi = VRPFprevphi + VRPFζ[i].ϕ[VRPF_index[1]-1]
            VRPFn += 1
        end
        BlockVRPF_index = findall(T[Block] .< BlockVRPFζ[i].τ .<= T[Block+1])
        if length(BlockVRPF_index) == NJumps
            BlockVRPFτ = BlockVRPFτ .+ BlockVRPFζ[i].τ[BlockVRPF_index]
            BlockVRPFϕ = BlockVRPFϕ .+ BlockVRPFζ[i].ϕ[BlockVRPF_index]
            BlockVRPFprevphi = BlockVRPFprevphi + BlockVRPFζ[i].ϕ[BlockVRPF_index[1]-1]
            BlockVRPFn += 1
        end
    end
    ResIndex = findall(T[Block] .< y.t .<= T[Block+1])
    JIndex = findall(T[Block] .< J.τ .<= T[Block+1])
    Jτ = [[T[Block]];J.τ[JIndex];[T[Block+1]]]
    Jϕ = J.ϕ[[[JIndex[1]-1];JIndex]]
    scatter(y.t[ResIndex],y.y[ResIndex],label="",color=:grey,markersize=4.0,markerstrodewidth=0.0)
    for i = 1:(length(Jτ)-1)
        plot!(Jτ[i:i+1],repeat([Jϕ[i]],2),label="",color=:red)
    end
    BR = (τ = [[T[Block]];BlockVRPFτ/BlockVRPFn;[T[Block+1]]],ϕ=[[BlockVRPFprevphi/BlockVRPFn];BlockVRPFϕ/BlockVRPFn])
    R = (τ = [[T[Block]];VRPFτ/VRPFn;[T[Block+1]]],ϕ=[[VRPFprevphi/VRPFn];VRPFϕ/VRPFn])
    for i = 1:(NJumps+1)
        plot!(BR.τ[i:i+1],repeat([BR.ϕ[i]],2),label="",color=:darkolivegreen)
        plot!(R.τ[i:i+1],repeat([R.ϕ[i]],2),label="",color=:orange)
    end
    current()
end
function EstimatedWholePDMP(NJumps::Int64,VRPFζ,BlockVRPFζ,T,J)
    VRPFτ = zeros(NJumps+1)
    VRPFϕ = zeros(NJumps+1)
    BlockVRPFτ = zeros(NJumps+1)
    BlockVRPFϕ = zeros(NJumps+1)
    VRPFn = 0
    BlockVRPFn = 0
    M = length(VRPFζ)
    for i = 1:M
        if VRPFζ[i].K == NJumps
            VRPFτ = VRPFτ .+ VRPFζ[i].τ
            VRPFϕ = VRPFϕ .+ VRPFζ[i].ϕ
            VRPFn += 1
        end
        if BlockVRPFζ[i].K == NJumps
            BlockVRPFτ = BlockVRPFτ .+ BlockVRPFζ[i].τ
            BlockVRPFϕ = BlockVRPFϕ .+ BlockVRPFζ[i].ϕ
            BlockVRPFn += 1
        end
    end
    ResIndex = findall(T[1] .< y.t .<= T[end])
    JIndex = findall(T[1] .<= J.τ .<= T[end])
    Jτ = [J.τ[JIndex];[T[end]]]
    Jϕ = J.ϕ[JIndex]
    scatter(y.t[ResIndex],y.y[ResIndex],label="",color=:grey,markersize=1.0,markerstrodewidth=0.0)
    for i = 1:(length(Jτ)-1)
        plot!(Jτ[i:i+1],repeat([Jϕ[i]],2),label="",color=:red)
    end
    BR = (τ = [BlockVRPFτ/BlockVRPFn;[T[end]]],ϕ=BlockVRPFϕ/BlockVRPFn)
    R = (τ = [VRPFτ/VRPFn;[T[end]]],ϕ=VRPFϕ/VRPFn)
    for i = 1:(NJumps+1)
        plot!(BR.τ[i:i+1],repeat([BR.ϕ[i]],2),label="",color=:darkolivegreen)
        plot!(R.τ[i:i+1],repeat([R.ϕ[i]],2),label="",color=:orange)
    end
    vline!(T,label="",color=:grey,linestyle=:dash,linewidth=2.0)
    current()
end


plot(EstimatedPDMP(1,3,VRPFζ,BlockVRPFζ,T,J),xlabel="Time",ylabel=L"$\phi$")
p1 = histogram(VRPFK[:,1],color=:orange,label="",xlabel="K");vline!([3],label="",color=:red,linestyle=:dash);
p2 = histogram(BlockVRPFK[:,1],color=:darkolivegreen,label="",xlabel="K");vline!([3],label="",color=:red,linestyle=:dash);
plot(p1,p2,layout=(1,2))

EstimatedPDMP(1,3,VRPFζ,BlockVRPFζ,T,J)

N = 100
VRPFζ = Array{Any}(undef,1000)
Random.seed!(1313)
R = VRPF.SMC(N,T,y,CP,par);
BR = VRPF.BS(R,y,T,CP,par)
VRPFζ[1] = BR.Path
L = BR.X
@showprogress 1 for i = 1:1000
    R = VRPF.cSMC(L,N,T,y,CP,par);
    BR = VRPF.BS(R,y,T,CP,par)
    VRPFζ[i] = BR.Path
    L = BR.X
end
BlockVRPFζ = Array{Any}(undef,1000)
Random.seed!(12345)
R = BlockVRPF.SMC(N,T,y,CP,par,[0.1,0.5]);
BR = BlockVRPF.BS(R,y,T,CP,par,[0.1,0.5])
BlockVRPFζ[1] = BR.Path
L = BR.L
@showprogress 1 for i = 1:1000
    R = BlockVRPF.cSMC(L,N,T,y,CP,par,[0.1,0.5]);
    BR = BlockVRPF.BS(R,y,T,CP,par,[0.1,0.5])
    BlockVRPFζ[i] = BR.Path
    L = BR.L
end
ToTVRPFK = zeros(Int64,1000)
ToTBlockVRPFK = zeros(Int64,1000)
for i = 1:1000
    ToTVRPFK[i] = VRPFζ[i].K
    ToTBlockVRPFK[i] = BlockVRPFζ[i].K 
end


LastJump_VRPF = zeros(1000,P-1)
LastJump_BlockVRPF = zeros(1000,P-1)
FirstJump_VRPF= zeros(1000,P-1)
FirstJump_BlockVRPF = zeros(1000,P-1)
for i = 1:1000
    for j = 2:P
        index = findlast(VRPFζ[i].τ .< T[j])
        if isnothing(index)
            LastJump_VRPF[i,j-1] = -Inf
        else
            LastJump_VRPF[i,j-1] = VRPFζ[i].τ[index]
        end
        index = findlast(BlockVRPFζ[i].τ .< T[j])
        if isnothing(index)
            LastJump_BlockVRPF[i,j-1] = -Inf
        else
            LastJump_BlockVRPF[i,j-1] = BlockVRPFζ[i].τ[index]
        end
        index = findfirst(VRPFζ[i].τ .> T[j])
        if isnothing(index)
            FirstJump_VRPF[i,j-1] = Inf
        else
            FirstJump_VRPF[i,j-1] = VRPFζ[i].τ[index]
        end
        index = findfirst(BlockVRPFζ[i].τ .> T[j])
        if isnothing(index)
            FirstJump_BlockVRPF[i,j-1] = Inf
        else
            FirstJump_BlockVRPF[i,j-1] = BlockVRPFζ[i].τ[index]
        end
    end
end

NearestJumpVRPF = zeros(1000,14)
NearestJumpBlockVRPF = zeros(1000,14)
for i = 1:1000
    for j = 1:14
        NearestJumpVRPF[i,j] = VRPFζ[i].τ[findmin(abs.(VRPFζ[i].τ .- J.τ[j+1]))[2]]
        NearestJumpBlockVRPF[i,j] = BlockVRPFζ[i].τ[findmin(abs.(BlockVRPFζ[i].τ .- J.τ[j+1]))[2]]
    end
end
histogram(LastJump_BlockVRPF[:,3],bins=50)
histogram(LastJump_VRPF[:,3],bins=50)