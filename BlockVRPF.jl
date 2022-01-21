module BlockVRPF
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
function SMC(N,TimeVec,y;model,par,auxpar)
    T = length(TimeVec) - 1
    Z = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    for i = 1:N
        X, SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
        Z[i,1] = model.Z(1,nothing,nothing,X)
        J[i,1] = Z[i,1].X
        W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
    end
    if isnan(findmax(W[:,1])[1])
        println(Z[findmax(W[:,1])[2],1])
        throw("Log Weights have NaN")
    elseif isinf(findmax(W[:,1])[1])
        println(Z[findmax(W[:,1])[2],1])
        throw("Log weights have inf")
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[:,n-1] = sample(1:N,Weights(NW[:,n-1]),N)
        for i = 1:N
            Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
            W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar,SampDenMat[i,n])
            J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
        end
        if isnan(findmax(W[:,n])[1])
            println(Z[findmax(W[:,n])[2],n])
            println(J[A[findmax(W[:,n])[2],n-1],n-1])
            throw("Log Weights have NaN")
        elseif isinf(findmax(W[:,n])[1])
            println(Z[findmax(W[:,n])[2],n])
            println(J[A[findmax(W[:,n])[2],n-1],n-1])
            throw("Log weights have inf")
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(Z,J,W,NW,A)
end
function cSMC(L,N,TimeVec,y;model,par,auxpar)
    T = length(TimeVec)-1
    Z = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    for i = 1:N
        if i == 1
            Z[i,1] = L[1]
            SampDenMat[i,1] = model.CalSampDen(Z[i,1].X,TimeVec[1],TimeVec[2],y,par)
            J[i,1] = Z[i,1].X
            W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        else
            X, SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            Z[i,1] = model.Z(1,nothing,nothing,X)
            J[i,1] = Z[i,1].X
            W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        end
    end
    if isnan(findmax(W[:,1])[1])
        println(Z[findmax(W[:,1])[2],1])
        throw("Log Weights have NaN")  
    elseif isinf(findmax(W[:,1])[1])
        println(Z[findmax(W[:,1])[2],1])
        throw("Log weights have inf")
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[:,n-1] = sample(1:N,Weights(NW[:,n-1]),N)
        A[1,n-1] = 1
        for i = 1:N
            if i == 1
                Z[i,n] = L[n]
                SampDenMat[i,n] = model.ProposedZDendity(Z[i,n],J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
                W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar,SampDenMat[i,n])
                J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
            else
                Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
                W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar,SampDenMat[i,n])
                J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
            end
        end
        if isnan(findmax(W[:,n])[1])
            println(Z[findmax(W[:,n])[2],n])
            println(J[A[findmax(W[:,n])[2],n-1],n-1])
            throw("Log Weights have NaN")
        elseif isinf(findmax(W[:,n])[1])
            println(Z[findmax(W[:,n])[2],n])
            println(J[A[findmax(W[:,n])[2],n-1],n-1])
            throw("Log weights have inf")
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(Z,J,W,NW,A)
end
function TraceLineage(A,n)
    T = size(A,2)+1
    output = zeros(Int64,T)
    output[T] = n
    for i = (T-1):-1:1
        output[i] = A[output[i+1],i]
    end
    return output
end
function BS(SMCR,y,TimeVec;model,par,auxpar)
    N,T = size(SMCR.Weights)
    BSWeight = zeros(N,T)
    BSWeight[:,T] = SMCR.NWeights[:,T]
    ParticleIndex = zeros(Int64,T)
    ParticleIndex[T] = sample(1:N,Weights(BSWeight[:,T]),1)[1]
    Laterξ = SMCR.Particles[ParticleIndex[T],T].X
    L = Vector{Any}(undef,T)
    L[T] = SMCR.Particles[ParticleIndex[T],T]
    for t = (T-1):-1:1
        for i = 1:N
            BSWeight[i,t] = SMCR.Weights[i,t]+model.BlockBSIncrementalWeight(SMCR.PDMP[i,t],L[t+1],Laterξ,y,TimeVec[t],TimeVec[t+1],TimeVec[end],par,auxpar)
        end
        BSWeight[:,t] = exp.(BSWeight[:,t] .- findmax(BSWeight[:,t])[1] )
        BSWeight[:,t] = BSWeight[:,t] / sum(BSWeight[:,t])
        ParticleIndex[t] = sample(1:N,Weights(BSWeight[:,t]),1)[1]
        L[t] = SMCR.Particles[ParticleIndex[t],t]
        if L[t+1].M == 1
            Laterξ = model.PDMP(L[t].X.K + 1 + Laterξ.K,[L[t].X.τ;[L[t+1].taum];Laterξ.τ],[L[t].X.ϕ;[L[t+1].phim];Laterξ.ϕ])
        else
            if isnothing(L[t+1].taum)
                Laterξ = model.PDMP(L[t].X.K + Laterξ.K,[L[t].X.τ;Laterξ.τ],[L[t].X.ϕ;Laterξ.ϕ])
            else
                Laterξ = model.PDMP(L[t].X.K + Laterξ.K,[L[t].X.τ[1:end-1];[L[t+1].taum];Laterξ.τ],[L[t].X.ϕ[1:end-1];[L[t+1].phim];Laterξ.ϕ])
            end
        end
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
    Globalalpha::Float64=0.3
    Componentalpha::Float64=0.5
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
function TunePars(model,y,T;method,auxpar,kws...)
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
    R = SMC(args.SMCAdaptN,args.T,y;model=model,par=oldpar,auxpar=auxpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
    Path = BSR.BackwardPath
    L = BSR.L
    L = model.Rejuvenate(Path,args.T,auxpar)
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
        if rand() < α
            oldpar = newpar
            oldθ = newθ
        end
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-10*I
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        """
        if method == "Component"
            println("lambda = ",λmat[n+1,:])
        else
            println("lambda = ",λvec[n+1])
        end
        """
        R = cSMC(L,args.SMCAdaptN,args.T,y,model=model,par=oldpar,auxpar=auxpar)
        BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
        Path = BSR.BackwardPath
        #push!(K,Path.K)
        #println("K=",Path.K,"llk=",model.JointDensity(Path,y,0.0,args.T[end],model.convert_to_pars(oldθ)))
        #println("No. Jumps = ",Path.K,"New Density = ",model.JointDensity(Path,y,0.0,args.T[end],oldpar))
        L = BSR.L
        L = model.Rejuvenate(Path,args.T,auxpar)
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
function PG(model,y,T;proppar=nothing,θ0=nothing,auxpar=[3.0,0.5],method="Global",kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    θ = zeros(args.NBurn+args.NChain+1,args.dim)
    if isnothing(proppar)
        λ,Σ,θ0 = TunePars(model,y,T;method=method,auxpar=auxpar,kws...)
    else
        λ,Σ = proppar
    end
    println("θ0 is",θ0)
    if isnothing(θ0)
        θ[1,:] = rand.(model.prior)
    else
        θ[1,:] = θ0
    end
    oldpar = model.convert_to_pars(θ[1,:])
    R = SMC(args.SMCN,args.T,y;model=model,par=oldpar,auxpar=auxpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
    Path = BSR.BackwardPath
    L = BSR.L
    L = model.Rejuvenate(Path,args.T,auxpar)
    @info "Running PG algorithms..."
    @showprogress 1 for n = 1:(args.NBurn+args.NChain)
        θ[n+1,:] = MH(θ[n,:],Path,y,args.NFold,method=method,model=model,T=T[end],λ=λ,Σ=Σ)
        R = cSMC(L,args.SMCN,args.T,y,model=model,par=model.convert_to_pars(θ[n+1,:]),auxpar=auxpar)
        BSR = BS(R,y,args.T,model=model,par=model.convert_to_pars(θ[n+1,:]),auxpar=auxpar)
        Path = BSR.BackwardPath
        L = BSR.L
        L = model.Rejuvenate(Path,args.T,auxpar)
    end
    return θ[(args.NBurn+2):end,:]
end
end
