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
    Later?? = SMCR.Particles[ParticleIndex[T],T].X
    L = Vector{Any}(undef,T)
    L[T] = SMCR.Particles[ParticleIndex[T],T]
    for t = (T-1):-1:1
        for i = 1:N
            BSWeight[i,t] = SMCR.Weights[i,t]+model.BlockBSIncrementalWeight(SMCR.PDMP[i,t],L[t+1],Later??,y,TimeVec[t],TimeVec[t+1],TimeVec[end],par,auxpar)
        end
        BSWeight[:,t] = exp.(BSWeight[:,t] .- findmax(BSWeight[:,t])[1])
        BSWeight[:,t] = BSWeight[:,t] / sum(BSWeight[:,t])
        if any(isnan.(BSWeight[:,t]))
            @save "error.jld2" BSWeight L Later?? par SMCR
        end
        ParticleIndex[t] = vcat(fill.(1:N,rand(Multinomial(1,BSWeight[:,t])))...)[1]
        L[t] = SMCR.Particles[ParticleIndex[t],t]
        if L[t+1].M == 1
            Later?? = model.PDMP(L[t].X.K + 1 + Later??.K,[L[t].X.??;[L[t+1].taum];Later??.??],[L[t].X.??;[L[t+1].phim];Later??.??])
        else
            if isnothing(L[t+1].taum)
                Later?? = model.PDMP(L[t].X.K + Later??.K,[L[t].X.??;Later??.??],[L[t].X.??;Later??.??])
            else
                Later?? = model.PDMP(L[t].X.K + Later??.K,[L[t].X.??[1:end-1];[L[t+1].taum];Later??.??],[L[t].X.??[1:end-1];[L[t+1].phim];Later??.??])
            end
        end
    end
    return BSRes(Later??,L,ParticleIndex)
end
@kwdef mutable struct PGargs
    dim::Int64
    ??0::Float64 = 1.0
    ??0::Matrix{Float64} = 10*Matrix{Float64}(I,dim,dim)
    ??0::Vector{Float64} = zeros(dim)
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
function LogPosteriorRatio(old??,new??,process,y,End,model)
    newprior = model.logprior(new??)
    oldprior = model.logprior(old??)
    newllk   = model.JointDensity(process,y,0.0,End,model.convert_to_pars(new??))
    oldllk   = model.JointDensity(process,y,0.0,End,model.convert_to_pars(old??))
    return newprior+newllk-oldprior-oldllk
end
function Tune??(old??,Z,??0,??;process,y,End,model,args)
    new?? =zeros(length(??0))
    k = length(??0)
    E = Matrix{Float64}(I,k,k)
    for i = 1:k
        tempnew?? = old?? .+ Z[i]*E[i,:]
        if model.logprior(tempnew??) == -Inf
            temp?? = 0.0
        else
            temp??    = min(1,exp(LogPosteriorRatio(old??,tempnew??,process,y,End,model)))
        end
        new??[i]  = exp(log(??0[i]) + ??*(temp??-args.Componentalpha))
    end
    return new??
end
function TunePars(model,y,T;??0=nothing,method,auxpar,kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    if method == "Component"
        ??mat = zeros(args.NAdapt+1,model.dim)
        ??mat[1,:] = args.??0 * ones(args.dim)
    else
        ??vec = zeros(args.NAdapt+1)
        ??vec[1] = args.??0
    end
    ??    = args.??0
    ??    = args.??0
    # initialise 
    if isnothing(??0)
        old?? = rand.(model.prior)
    else
        old?? =??0
    end
    oldpar = model.convert_to_pars(old??)
    R = SMC(args.SMCAdaptN,args.T,y;model=model,par=oldpar,auxpar=auxpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
    Path = BSR.BackwardPath
    # update
    @info "Tuning PG parameters..."
    @showprogress 1 for n = 1:args.NAdapt
        # Propose new parameters
        if method == "Component"
            ?? = Matrix{Float64}(Diagonal(sqrt.(??mat[n,:])))
            vars = ??*cholesky(??).L; vars = vars*transpose(vars)
            Z = rand(MultivariateNormal(zeros(args.dim),vars))
        else
            Z = rand(MultivariateNormal(zeros(args.dim),??vec[n]*??))
        end
        new?? = old??.+ Z
        newpar = model.convert_to_pars(new??)
        if model.logprior(new??) > -Inf
            LLkRatio = LogPosteriorRatio(old??,new??,Path,y,args.T[end],model)
            ?? = min(1,exp(LLkRatio))
        else
            ?? = 0.0
        end
        if method=="Component"
            ??mat[n+1,:] =Tune??(old??,Z,??mat[n,:],n^(-1/3),process=Path,y=y,End=args.T[end],model=model,args=args)
        else
            ??vec[n+1] = exp(log(??vec[n])+n^(-1/3)*(?? - args.Globalalpha))
        end
        if rand() < ??
            oldpar = newpar
            old?? = new??
        end
        println(old??)
        ?? = ?? + n^(-1/3)*((old??.-??)*transpose(old??.-??)-??)+1e-10*I
        ?? = ?? .+ n^(-1/3)*(old?? .- ??)
        L = model.Rejuvenate(Path,args.T,auxpar,oldpar)
        R = cSMC(L,args.SMCAdaptN,args.T,y,model=model,par=oldpar,auxpar=auxpar)
        BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
        Path = BSR.BackwardPath
        #L = model.Rejuvenate(Path,args.T,auxpar)
    end
    if method == "Component"
        return (??mat[end,:],??,old??)
    else
        return (??vec[end],??,old??)
    end
end
function MH(??0,Process,y,NIter;method,model,T,??,??)
    # Create Output Arrray
    if method == "Component"
        ?? = Matrix{Float64}(Diagonal(sqrt.(??)))
        vars = ??*cholesky(??).L; vars = vars*transpose(vars)
    else
        vars = ??*??
    end
    old?? = ??0
    oldpar = model.convert_to_pars(??0)
    llk0 = model.JointDensity(Process,y,0.0,T,oldpar)
    prior0 = model.logprior(old??)
    for n = 1:NIter
        new?? = rand(MultivariateNormal(old??,vars))
        prior1 = model.logprior(new??)
        if prior1 == -Inf
            ?? = 0.0
        else
            llk1 = model.JointDensity(Process,y,0.0,T,model.convert_to_pars(new??))
            ?? = min(1,exp(llk1+prior1-llk0-prior0))
        end
        if rand() < ??
            old?? = new??
            llk0 = llk1
            prior0 = prior1
        end
    end
    return old??
end
function PG(model,y,T;proppar=nothing,??0=nothing,auxpar=[3.0,0.5],method="Global",kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    ?? = zeros(args.NBurn+args.NChain+1,args.dim)
    if isnothing(proppar)
        ??,??,??0 = TunePars(model,y,T;??0=??0,method=method,auxpar=auxpar,kws...)
    else
        ??,?? = proppar
    end
    println("??0 is",??0)
    println("?? is $??")
    println("?? is ")
    display(??)
    if isnothing(??0)
        ??[1,:] = rand.(model.prior)
    else
        ??[1,:] = ??0
    end
    oldpar = model.convert_to_pars(??[1,:])
    R = SMC(args.SMCN,args.T,y;model=model,par=oldpar,auxpar=auxpar)
    BSR = BS(R,y,args.T,model=model,par=oldpar,auxpar=auxpar)
    Path = BSR.BackwardPath
    L = model.Rejuvenate(Path,args.T,auxpar)
    @info "Running PG algorithms with BlockVRPF method..."
    @showprogress 1 for n = 1:(args.NBurn+args.NChain)
        ??[n+1,:] = MH(??[n,:],Path,y,args.NFold,method=method,model=model,T=args.T[end],??=??,??=??)
        R = cSMC(L,args.SMCN,args.T,y,model=model,par=model.convert_to_pars(??[n+1,:]),auxpar=auxpar)
        BSR = BS(R,y,args.T,model=model,par=model.convert_to_pars(??[n+1,:]),auxpar=auxpar)
        Path = BSR.BackwardPath
        L = model.Rejuvenate(Path,args.T,auxpar)
    end
    return ??[(args.NBurn+2):end,:]
end
end
