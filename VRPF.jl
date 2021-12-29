module VRPF
export SMC
using Distributions, StatsBase, Random
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
        W[i,1] = model.DensityRatio(X[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        J[i,1] = X[i,1]
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[:,n-1] = sample(1:N,Weights(NW[:,n-1]),N)
        for i = 1:N
            X[i,n],SampDenMat[i,n] = model.GenParticle(TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
            W[i,n] = model.DensityRatio(X[i,n],J[A[i,n-1],n-1],y,TimeVec[n],TimeVec[n+1],par) - SampDenMat[i,n]
            J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
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
            W[i,1] = model.DensityRatio(X[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        else
            X[i,1],SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            W[i,1] = model.DensityRatio(X[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
            J[i,1] = X[i,1]
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
                W[i,n] =  model.DensityRatio(X[i,n],J[A[i,n-1],n-1],y,TimeVec[n],TimeVec[n+1],par) - SampDenMat[i,n]
                J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
            else
                X[i,n],SampDenMat[i,n] = model.GenParticle(TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
                W[i,n] = model.DensityRatio(X[i,n],J[A[i,n-1],n-1],y,TimeVec[n],TimeVec[n+1],par) - SampDenMat[i,n]
                J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
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
            α = exp(min(0,sum(logpdf.(model.prior,newθ))+model.CalPDMP(Path,args.T[end],newpar)-sum(logpdf.(model.prior,oldθ))-model.CalPDMP(Path,args.T[end],oldpar)))
        else
            α = 0.0
        end
        if rand() < α
            oldpar = newpar
            oldθ = newθ
        end
        println(oldθ)
        λvec[n+1] = exp(log(λvec[n])+n^(-1/4)*(α-0.234))
        #println(size((oldθ.-μ)*transpose(oldθ.-μ)))
        Σ = Σ + n^(-1/4)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-3*I
        μ = μ .+ n^(-1/4)*(oldθ .- μ)
        R = VRPF.cSMC(L,args.SMCN,args.T,y,model=model,par=oldpar)
        BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
        Path = BSR.BackwardPath
        L = BSR.L
    end
    return (λvec[end],Σ,oldθ)
end
end