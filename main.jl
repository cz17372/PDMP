using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra
include("ChangePoint.jl")
using Base:@kwdef
using .ChangePoint
include("VRPF.jl")
ξ,y = ChangePoint.SimData(seed=17370)
ChangePoint.ProcObsGraph(ξ,y)
model = ChangePoint
@kwdef mutable struct PGargs
    dim::Int64
    λ0::Float64 = 1.0
    Σ0::Matrix{Float64} = Matrix{Float64}(I,dim,dim)
    μ0::Vector{Float64} = zeros(dim)
    NAdapt::Int64 = 50000
    NBurn::Int64 = 10000
    NChain::Int64= 100000
    SMCN::Int64 = 100
    T::Vector{Float64}
end
function TunePars(model,y;args)
    args = args
    λvec = zeros(args.NAdapt+1)
    Σ    = args.Σ0
    μ    = args.μ0
    λvec[1] = args.λ0
    # initialise 
    oldθ = rand.(model.prior)
    oldpar = ChangePoint.pars(ρ=oldθ[1],σϕ=oldθ[2],σy=oldθ[3],α=oldθ[4],β=oldθ[5])
    R = VRPF.SMC(args.SMCN,args.T,y;model=model,par=oldpar)
    BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
    Path = BSR.BackwardPath
    L = BSR.L
    # update
    for n = 1:args.NAdapt
        newθ = rand(MultivariateNormal(oldθ,λvec[n]*Σ))
        newpar = ChangePoint.pars(ρ=newθ[1],σϕ=newθ[2],σy=newθ[3],α=newθ[4],β=newθ[5])
        if sum(logpdf.(model.prior,newθ)) > -Inf
            α = exp(min(0,sum(logpdf.(model.prior,newθ))+model.CalPDMP(Path,args.T[end],newpar)+model.CalLlk(Path,y,0.0,args.T[end],newpar)-sum(logpdf.(model.prior,oldθ))-model.CalPDMP(Path,args.T[end],oldpar)-model.CalLlk(Path,y,0.0,args.T[end],oldpar)))
        else
            α = 0.0
        end
        if rand() < α
            oldpar = newpar
            oldθ = newθ
        end
        println(oldθ)
        λvec[n+1] = exp(log(λvec[n])+n^(-1/3)*(α-0.234))
        #println(size((oldθ.-μ)*transpose(oldθ.-μ)))
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-3*I
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        R = VRPF.cSMC(L,args.SMCN,args.T,y,model=model,par=oldpar)
        BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
        Path = BSR.BackwardPath
        L = BSR.L
    end
    return (λvec[end],Σ,oldθ,Path)
end
R = VRPF.SMC(10000,collect(0:100:1000),y,model=ChangePoint,par=ChangePoint.pars())
model=ChangePoint
args=PGargs(dim=5,T=collect(0.0:100.0:1000.0))

R = TunePars(model,y,args=args)
model.ProcObsGraph(R[4],y)
