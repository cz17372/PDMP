module VRPF
export SMC
using Distributions, StatsBase, Random, LinearAlgebra, ProgressMeter
using Base:@kwdef
mutable struct SMCRes
    Particles::Matrix{Any}
    PDMP::Matrix{Any}
    Weights::Matrix{Float64}
    NWeights::Matrix{Float64}
    Ancestor::Matrix{Int64}
end
mutable struct BSRes
    BackwardPath
    L::Vector{Any}
    BackIndex::Vector{Int64}
end
function SMC(N,TimeVec,y;model,par)
    T = length(TimeVec)-1
    X = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    # Sample the particles in the first time block
    for i = 1:N
        X[i,1],SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
        J[i,1] = X[i,1]
        W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]  
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[:,n-1] = sample(1:N,Weights(NW[:,n-1]),N)
        for i = 1:N
            X[i,n],SampDenMat[i,n] = model.GenParticle(TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
            J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
            W[i,n] = model.JointDensityRatio(J[A[i,n-1],n-1],TimeVec[n],J[i,n],TimeVec[n+1],y,par) - SampDenMat[i,n]
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(X,J,W,NW,A)
end
function cSMC(L,N,TimeVec,y;model,par)
    T = length(TimeVec)-1
    X = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    # Sample the particles in the first time block
    for i = 1:N
        if i == 1
            X[i,1] = L[1]
            SampDenMat[i,1] = model.CalSampDen(X[i,1],TimeVec[1],TimeVec[2],y,par)
            J[i,1] = X[i,1]
            W[i,1] =  model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]  
        else
            X[i,1],SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            J[i,1] = X[i,1]
            W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        end
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[1,n-1] = 1
        A[2:N,n-1] = sample(1:N,Weights(NW[:,n-1]),N-1)
        for i = 1:N
            if i == 1
                X[i,n] = L[n]
                SampDenMat[i,n] = model.CalSampDen(X[i,n],TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
                J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
                W[i,n] = model.JointDensityRatio(J[A[i,n-1],n-1],TimeVec[n],J[i,n],TimeVec[n+1],y,par) - SampDenMat[i,n]
            else
                X[i,n],SampDenMat[i,n] = model.GenParticle(TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
                J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
                W[i,n] = model.JointDensityRatio(J[A[i,n-1],n-1],TimeVec[n],J[i,n],TimeVec[n+1],y,par) - SampDenMat[i,n]
            end
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(X,J,W,NW,A)
end
function BS(SMCR,y,TimeVec;model,par)
    N,T = size(SMCR.Weights)
    BSWeight = zeros(N,T)
    BSWeight[:,T] = SMCR.NWeights[:,T]
    ParticleIndex = zeros(Int64,T)
    ParticleIndex[T] = sample(1:N,Weights(BSWeight[:,T]),1)[1]
    Laterξ = SMCR.Particles[ParticleIndex[T],T]
    L = Vector{Any}(undef,T)
    L[T] = SMCR.Particles[ParticleIndex[T],T]
    for t = (T-1):-1:1
        for i = 1:N
            BSWeight[i,t] = SMCR.Weights[i,t]+model.BSRatio(SMCR.PDMP[i,t],Laterξ,y,TimeVec[t],TimeVec[t+1],TimeVec[end],par)
        end
        BSWeight[:,t] = exp.(BSWeight[:,t] .- findmax(BSWeight[:,t])[1] )
        BSWeight[:,t] = BSWeight[:,t] / sum(BSWeight[:,t])
        ParticleIndex[t] = sample(1:N,Weights(BSWeight[:,t]),1)[1]
        L[t] = SMCR.Particles[ParticleIndex[t],t]
        Laterξ = model.insertPDMP(Laterξ,L[t])
    end
    return BSRes(Laterξ,L,ParticleIndex)
end
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
function TunePars(model,y,T;kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    λvec = zeros(args.NAdapt+1)
    Σ    = args.Σ0
    μ    = args.μ0
    λvec[1] = args.λ0
    # initialise 
    oldθ = rand.(model.prior)
    oldpar = model.convert_to_pars(oldθ)
    R = VRPF.SMC(args.SMCN,args.T,y;model=model,par=oldpar)
    BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
    Path = BSR.BackwardPath
    L = BSR.L
    # update
    @info "Tuning PG parameters..."
    for n = 1:args.NAdapt
        newθ = rand(MultivariateNormal(oldθ,λvec[n]*Σ))
        newpar = model.convert_to_pars(newθ)
        if sum(logpdf.(model.prior,newθ)) > -Inf
            α = exp(min(0,sum(logpdf.(model.prior,newθ))+model.JointDensity(Path,y,0.0,args.T[end],newpar)-sum(logpdf.(model.prior,oldθ))-model.JointDensity(Path,y,0.0,args.T[end],oldpar)))
        else
            α = 0.0
        end
        if rand() < α
            oldpar = newpar
            oldθ = newθ
        end
        println(oldθ)
        λvec[n+1] = exp(log(λvec[n])+n^(-1/3)*(α-0.234))
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        #println(size((oldθ.-μ)*transpose(oldθ.-μ)))
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-10*I
        println("lambda = ",λvec[n+1])
        R = VRPF.cSMC(L,args.SMCN,args.T,y,model=model,par=oldpar)
        BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
        Path = BSR.BackwardPath
        println("No. Jumps = ",Path.K)
        L = BSR.L
        println("Likelihood of observations is",model.CalLlk(Path,y,0.0,args.T[end],oldpar))
    end
    return (λvec[end],Σ)
end
function PG(model,y,T;kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    θ = zeros(args.NBurn+args.NChain+1,args.dim)
    λ,Σ = TunePars(model,y,T;kws...)
    θ[1,:] = rand.(model.prior)
    oldpar = model.convert_to_pars(θ[1,:])
    R = VRPF.SMC(args.SMCN,args.T,y;model=model,par=oldpar)
    BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
    Path = BSR.BackwardPath
    L = BSR.L
    @info "Running PG algorithms..."
    @showprogress 1 for n = 1:(args.NBurn+args.NChain)
        newθ = rand(MultivariateNormal(θ[n,:],λ*Σ))
        newpar = model.convert_to_pars(newθ)
        if sum(logpdf.(model.prior,newθ)) > -Inf
            α = exp(min(0,sum(logpdf.(model.prior,newθ))+model.CalPDMP(Path,args.T[end],newpar)+model.CalLlk(Path,y,0.0,args.T[end],newpar)-sum(logpdf.(model.prior,θ[n,:]))-model.CalPDMP(Path,args.T[end],oldpar)-model.CalLlk(Path,y,0.0,args.T[end],oldpar)))
        else
            α = 0.0
        end
        if rand() < α
            oldpar = newpar
            θ[n+1,:] = newθ
        else
            θ[n+1,:] = θ[n,:]
        end
        R = VRPF.cSMC(L,args.SMCN,args.T,y,model=model,par=oldpar)
        BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
        Path = BSR.BackwardPath
        L = BSR.L
    end
    return θ[(args.NBurn+2):end,:]
end

end