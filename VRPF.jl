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
    Σ0::Matrix{Float64} = 10.0*Matrix{Float64}(I,dim,dim)
    μ0::Vector{Float64} = zeros(dim)
    NAdapt::Int64 = 50000
    NBurn::Int64 = 10000
    NChain::Int64= 50000
    SMCAdaptN::Int64 = 10
    SMCN::Int64 = 100
    T::Vector{Float64}
    NFold::Int64 = 500
    Globalalpha::Float64 = 0.25
    Componentalpha::Float64 = 0.5
end
function LogPosteriorRatio(oldθ,newθ,process,y,End,model)
    newprior = model.logprior(newθ)
    oldprior = model.logprior(oldθ)
    newllk   = model.JointDensity(process,y,0.0,End,model.convert_to_pars(newθ))
    oldllk   = model.JointDensity(process,y,0.0,End,model.convert_to_pars(oldθ))
    return newprior+newllk-oldprior-oldllk
end
function Tuneλ(oldθ,Z,λ0,γ;process,y,End,model,args)
    newλ =zeros(length(λ0))
    k = length(λ0)
    E = Matrix{Float64}(I,k,k)
    for i = 1:k
        tempnewθ = oldθ .+ Z[i]*E[i,:]
        if model.logprior(tempnewθ) == -Inf
            tempα = 0.0
        else
            tempα    = min(1,exp(LogPosteriorRatio(oldθ,tempnewθ,process,y,End,model)))
        end
        newλ[i]  = exp(log(λ0[i]) + γ*(tempα-args.Componentalpha))
    end
    return newλ
end
function TunePars(model,y,T;method,kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    if method == "Component"
        λmat = zeros(args.NAdapt+1,model.dim)
        λmat[1,:] = args.λ0 * ones(args.dim)
    else
        λvec = zeros(args.NAdapt+1)
        λvec[1] = args.λ0
    end
    Σ    = args.Σ0
    μ    = args.μ0
    
    # initialise 
    oldθ = rand.(model.prior)
    #oldθ = [0.0,2.0,2.0,10.0,10.0]
    oldpar = model.convert_to_pars(oldθ)
    R = SMC(args.SMCAdaptN,args.T,y;model=model,par=oldpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar)
    Path = BSR.BackwardPath
    L = BSR.L
    # update
    @info "Tuning PG parameters..."
    @showprogress 1 for n = 1:args.NAdapt
        # Propose new parameters
        if method == "Component"
            Λ = Matrix{Float64}(Diagonal(sqrt.(λmat[n,:])))
            vars = Λ*cholesky(Σ).L; vars = vars*transpose(vars)
            Z = rand(MultivariateNormal(zeros(args.dim),vars))
        else
            Z = rand(MultivariateNormal(zeros(args.dim),λvec[n]*Σ))
        end
        newθ = oldθ.+Z
        newpar = model.convert_to_pars(newθ)
        if model.logprior(newθ) > -Inf
            LLkRatio = LogPosteriorRatio(oldθ,newθ,Path,y,args.T[end],model)
            α = min(1,exp(LLkRatio))
            #println("density of old is",model.JointDensity(Path,y,0.0,args.T[end],oldpar))
            #println("density of new is",model.JointDensity(Path,y,0.0,args.T[end],newpar))
        else
            α = 0.0
        end
        if method=="Component"
            λmat[n+1,:] =Tuneλ(oldθ,Z,λmat[n,:],n^(-1/3),process=Path,y=y,End=args.T[end],model=model,args=args)
        else
            λvec[n+1] = exp(log(λvec[n])+n^(-1/3)*(α - args.Globalalpha))
        end
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-10*I
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        if rand() < α
            oldpar = newpar
            oldθ = newθ
        end
        #println(oldθ)
        """
        if method == "Component"
            println("lambda = ",λmat[n+1,:])
        else
            println("lambda = ",λvec[n+1])
        end
        """
        R = cSMC(L,args.SMCAdaptN,args.T,y,model=model,par=oldpar)
        BSR = BS(R,y,args.T,model=model,par=oldpar)
        Path = BSR.BackwardPath
        #println("K=",Path.K,"llk=",model.JointDensity(Path,y,0.0,args.T[end],model.convert_to_pars(oldθ)))
        #println("No. Jumps = ",Path.K,"New Density = ",model.JointDensity(Path,y,0.0,args.T[end],oldpar))
        L = BSR.L
    end
    if method == "Component"
        return (λmat[end,:],Σ,oldθ)
    else
        return (λvec[end],Σ,oldθ)
    end
end
function MH(θ0,Process,y,NIter;method,model,T,λ,Σ)
    # Create Output Arrray
    if method == "Component"
        Λ = Matrix{Float64}(Diagonal(sqrt.(λ)))
        vars = Λ*cholesky(Σ).L; vars = vars*transpose(vars)
    else
        vars = λ*Σ
    end
    oldθ = θ0
    oldpar = model.convert_to_pars(θ0)
    llk0 = model.JointDensity(Process,y,0.0,T,oldpar)
    prior0 = model.logprior(oldθ)
    for n = 1:NIter
        newθ = rand(MultivariateNormal(oldθ,vars))
        prior1 = model.logprior(newθ)
        if prior1 == -Inf
            α = 0.0
        else
            llk1 = model.JointDensity(Process,y,0.0,T,model.convert_to_pars(newθ))
            α = min(1,exp(llk1+prior1-llk0-prior0))
        end
        if rand() < α
            oldθ = newθ
            llk0 = llk1
            prior0 = prior1
        end
    end
    return oldθ
end
function PG(model,y,T;proppar=nothing,θ0=nothing,method="Global",kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    θ = zeros(args.NBurn+args.NChain+1,args.dim)
    if isnothing(proppar)
        λ,Σ,θ0 = TunePars(model,y,T;method=method,kws...)
    else
        λ,Σ = proppar
    end
    if isnothing(θ0)
        θ[1,:] = rand.(model.prior)
    else
        θ[1,:] = θ0
    end
    oldpar = model.convert_to_pars(θ[1,:])
    R = VRPF.SMC(args.SMCN,args.T,y;model=model,par=oldpar)
    BSR = VRPF.BS(R,y,args.T,model=model,par=oldpar)
    Path = BSR.BackwardPath
    L = BSR.L
    @info "Running PG algorithms..."
    @showprogress 1 for n = 1:(args.NBurn+args.NChain)
        θ[n+1,:] = MH(θ[n,:],Path,y,args.NFold,method=method,model=model,T=T[end],λ=λ,Σ=Σ)
        R = VRPF.cSMC(L,args.SMCN,args.T,y,model=model,par=model.convert_to_pars(θ[n+1,:]))
        BSR = VRPF.BS(R,y,args.T,model=model,par=model.convert_to_pars(θ[n+1,:]))
        Path = BSR.BackwardPath
        L = BSR.L
    end
    return θ[(args.NBurn+2):end,:]
end
end