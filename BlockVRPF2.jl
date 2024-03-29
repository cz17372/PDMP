module BlockVRPF
using Distributions, StatsBase, Random, LinearAlgebra, ProgressMeter,JLD2
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
        while isinf(SampDenMat[i,1]) | isnan(SampDenMat[i,1])
            X, SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
        end
        Z[i,1] = model.Z(1,nothing,nothing,X)
        J[i,1] = Z[i,1].X
        W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
    end
    MAX,ind = findmax(W[:,1])
    NW[:,1] = exp.(W[:,1] .- MAX)
    NW[:,1] = NW[:,1]/sum(NW[:,1])
    for n = 2:T
        A[:,n-1] = vcat(fill.(1:N,rand(Multinomial(N,NW[:,n-1])))...)
        for i = 1:N
            Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
            while isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
            end
            W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar,SampDenMat[i,n])
            J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
        end
        MAX,ind = findmax(W[:,n])
        NW[:,n] = exp.(W[:,n] .- MAX)
        NW[:,n] = NW[:,n]/sum(NW[:,n])
        if any(isnan.(NW[:,n]))
            @save "error.jld2" J Z A W par auxpar NW SampDenMat
        end
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
            if isinf(SampDenMat[i,1]) | isnan(SampDenMat[i,1])
                W[i,1] = -Inf
            else
                W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
            end
        else
            X, SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            while isinf(SampDenMat[i,1]) | isnan(SampDenMat[i,1])
                X, SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            end
            Z[i,1] = model.Z(1,nothing,nothing,X)
            J[i,1] = Z[i,1].X
            if isinf(SampDenMat[i,1])
                W[i,1] = -Inf
            else
                W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
            end
        end
    end
    MAX,ind = findmax(W[:,1])
    NW[:,1] = exp.(W[:,1] .- MAX)
    NW[:,1] = NW[:,1]/sum(NW[:,1])
    for n = 2:T
        A[2:end,n-1] = vcat(fill.(1:N,rand(Multinomial(N-1,NW[:,n-1])))...)
        A[1,n-1] = 1
        for i = 1:N
            if i == 1
                Z[i,n] = L[n]
                SampDenMat[i,n] = model.ProposedZDendity(Z[i,n],J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
                if isinf(SampDenMat[i,n]) | any(isinf.(W[i,1:n-1])) | isnan(SampDenMat[i,n])
                    W[i,n] = -Inf
                else
                    W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar,SampDenMat[i,n])
                end
                J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
            else
                Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
                while isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                    Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar)
                end
                if isinf(SampDenMat[i,n])
                    W[i,n] == -Inf
                else
                    W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,auxpar,SampDenMat[i,n])
                end
                J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
            end
        end
        MAX,ind = findmax(W[:,n])
        NW[:,n] = exp.(W[:,n] .- MAX)
        NW[:,n] = NW[:,n]/sum(NW[:,n])
        if any(isnan.(NW[:,n]))
            @save "error.jld2" J Z A W par auxpar L SampDenMat NW n
        end
    end
    return SMCRes(Z,J,W,NW,A)
end
function BS(SMCR,y,TimeVec;model,par,auxpar)
    N,T = size(SMCR.Weights)
    BSWeight = zeros(N,T)
    BSWeight[:,T] = SMCR.NWeights[:,T]
    if any(isnan.(BSWeight[:,T]))
        @save "error.jld2" SMCR par
    end
    ParticleIndex = zeros(Int64,T)
    ParticleIndex[T] = vcat(fill.(1:N,rand(Multinomial(1,SMCR.NWeights[:,T])))...)[1]
    Laterξ = SMCR.Particles[ParticleIndex[T],T].X
    L = Vector{Any}(undef,T)
    L[T] = SMCR.Particles[ParticleIndex[T],T]
    for t = (T-1):-1:1
        for i = 1:N
            BSWeight[i,t] = SMCR.Weights[i,t]+model.BlockBSIncrementalWeight(SMCR.PDMP[i,t],L[t+1],Laterξ,y,TimeVec[t],TimeVec[t+1],TimeVec[end],par,auxpar)
        end
        BSWeight[:,t] = exp.(BSWeight[:,t] .- findmax(BSWeight[:,t])[1])
        BSWeight[:,t] = BSWeight[:,t] / sum(BSWeight[:,t])
        if any(isnan.(BSWeight[:,t]))
            @save "error.jld2" BSWeight L Laterξ par SMCR
        end
        ParticleIndex[t] = vcat(fill.(1:N,rand(Multinomial(1,BSWeight[:,t])))...)[1]
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
    Σ0::Matrix{Float64} = 10*Matrix{Float64}(I,dim,dim)
    μ0::Vector{Float64} = zeros(dim)
    NAdapt::Int64 = 50000
    NBurn::Int64 = 20000
    NChain::Int64= 50000
    SMCAdaptN::Int64 = 10
    SMCN::Int64 = 100
    T::Vector{Float64}
    NFold::Int64 = 500
    Globalalpha::Float64=0.25
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
function TunePars(model,y,T;θ0=nothing,method,auxpar,kws...)
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
    if isnothing(θ0)
        oldθ = rand.(model.prior)
    else
        oldθ =θ0
    end
    oldpar = model.convert_to_pars(oldθ)
    R = SMC(args.SMCAdaptN,args.T,y;model=model,par=oldpar,auxpar=auxpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
    Path = BSR.BackwardPath
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
        newθ = oldθ.+ Z
        newpar = model.convert_to_pars(newθ)
        if model.logprior(newθ) > -Inf
            LLkRatio = LogPosteriorRatio(oldθ,newθ,Path,y,args.T[end],model)
            α = min(1,exp(LLkRatio))
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
        println(oldθ)
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-10*I
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        L = model.Rejuvenate(Path,args.T,auxpar,oldpar)
        R = cSMC(L,args.SMCAdaptN,args.T,y,model=model,par=oldpar,auxpar=auxpar)
        BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
        Path = BSR.BackwardPath
        #L = model.Rejuvenate(Path,args.T,auxpar)
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
        λ,Σ,θ0 = TunePars(model,y,T;θ0=θ0,method=method,auxpar=auxpar,kws...)
    else
        λ,Σ = proppar
    end
    println("θ0 is",θ0)
    println("λ is $λ")
    println("Σ is ")
    display(Σ)
    if isnothing(θ0)
        θ[1,:] = rand.(model.prior)
    else
        θ[1,:] = θ0
    end
    oldpar = model.convert_to_pars(θ[1,:])
    R = SMC(args.SMCN,args.T,y;model=model,par=oldpar,auxpar=auxpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
    Path = BSR.BackwardPath
    L = model.Rejuvenate(Path,args.T,auxpar)
    @info "Running PG algorithms with BlockVRPF method..."
    @showprogress 1 for n = 1:(args.NBurn+args.NChain)
        θ[n+1,:] = MH(θ[n,:],Path,y,args.NFold,method=method,model=model,T=args.T[end],λ=λ,Σ=Σ)
        R = cSMC(L,args.SMCN,args.T,y,model=model,par=model.convert_to_pars(θ[n+1,:]),auxpar=auxpar)
        BSR = BS(R,y,args.T,model=model,par=model.convert_to_pars(θ[n+1,:]),auxpar=auxpar)
        Path = BSR.BackwardPath
        L = model.Rejuvenate(Path,args.T,auxpar)
    end
    return θ[(args.NBurn+2):end,:]
end
end
