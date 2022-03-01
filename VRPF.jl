module VRPF
using Distributions, StatsBase, Statistics, Plots, JLD2, ProgressMeter,LinearAlgebra
using Base:@kwdef
function log_pdmp_Prior(J,EndTime,model,par)
    # Joint density for the jump times
    log_τ_density = sum(model.logf.(J.τ[2:end],J.τ[1:end-1],Ref(par)))
    log_ϕ_density = model.logϕ0(J.ϕ[1],par) + sum(model.logg.(J.ϕ[2:end],J.ϕ[1:end-1],J.τ[1:end-1],J.τ[2:end],Ref(par)))
    log_S_density = model.logS(J.τ[end],EndTime,par)
    return log_τ_density + log_ϕ_density + log_S_density
end
function log_pdmp_posterior(J,EndTime,y,model,par)
    logprior = log_pdmp_Prior(J,EndTime,model,par)
    llk      = model.CalLlk(J,y,0.0,EndTime,par)
    return llk + logprior
end
mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end
function KX(t1,y,model,par)
    llk = 0.0
    KDist = model.K_K(0.0,t1,y,par)
    K = rand(KDist)
    llk += logpdf(KDist,K)
    τ = sort(rand(Uniform(0.0,t1),K))
    llk += sum(log.(collect(1:K)/(t1-0)))
    if K == 0
        ϕ0Dist = model.K_ϕ0(t1,y,par)
        ϕ0 = rand(ϕ0Dist)
        llk += logpdf(ϕ0Dist,ϕ0)
        return (PDMP(K,[0.0],[ϕ0]),llk)
    else
        ϕ0Dist = model.K_ϕ0(τ[1],y,par)
        ϕ0 = rand(ϕ0Dist)
        llk += logpdf(ϕ0Dist,ϕ0)
        extendedτ = [[0.0];τ;[t1]]
        ϕ = [[ϕ0];zeros(K)]
        for n = 2:(K+1)
            ϕDist = model.K_ϕ(extendedτ[n-1],ϕ[n-1],extendedτ[n],extendedτ[n+1],y,par)
            ϕ[n] = rand(ϕDist)
            llk += logpdf(ϕDist,ϕ[n])
        end
        if any(isinf.(ϕ)) | isinf(llk) | isnan(llk)
            return KX(t1,y,model,par)
        else
            return (PDMP(K,[[0.0];τ],ϕ),llk)
        end
    end
end
function KX(t0,t1,J0,y,model,par)
    llk = 0.0
    KDist = model.K_K(t0,t1,y,par)
    K = rand(KDist)
    llk += logpdf(KDist,K)
    τ = sort(rand(Uniform(t0,t1),K))
    llk += sum(log.(collect(1:K)/(t1-t0)))
    ϕ = zeros(K)
    extendedτ = [τ;[t1]]
    prevtau = J0.τ[end]
    prevphi = J0.ϕ[end]
    for n = 1:K
        ϕDist = model.K_ϕ(prevtau,prevphi,extendedτ[n],extendedτ[n+1],y,par)
        ϕ[n] = rand(ϕDist)
        llk += logpdf(ϕDist,ϕ[n])
        prevtau = τ[n]
        prevphi = ϕ[n]
    end
    if any(isinf.(ϕ)) | isinf(llk) | isnan(llk)
        return KX(t0,t1,J0,y,model,par)
    else
        return (PDMP(K,τ,ϕ),llk)
    end
    
end
function qX(X,t1,y,model,par)
    llk = 0.0
    KDist = model.K_K(0.0,t1,y,par)
    K = rand(KDist)
    llk += logpdf(KDist,X.K)
    llk += sum(log.(collect(1:X.K)/(t1-0)))
    if X.K == 0
        ϕ0Dist = model.K_ϕ0(t1,y,par)
        llk += logpdf(ϕ0Dist,X.ϕ[1])
    else
        ϕ0Dist = model.K_ϕ0(X.τ[2],y,par)
        llk += logpdf(ϕ0Dist,X.ϕ[1])
        extendedτ = [X.τ;[t1]]
        for n = 2:(X.K+1)
            ϕDist = model.K_ϕ(extendedτ[n-1],X.ϕ[n-1],extendedτ[n],extendedτ[n+1],y,par)
            llk += logpdf(ϕDist,X.ϕ[n])
        end
    end
    return llk
end
function qX(X,t0,t1,J0,y,model,par)
    llk = 0.0
    KDist = model.K_K(t0,t1,y,par)
    llk += logpdf(KDist,X.K)
    llk += sum(log.(collect(1:X.K)/(t1-t0)))
    extendedτ = [X.τ;[t1]]
    prevtau = J0.τ[end]
    prevphi = J0.ϕ[end]
    for n = 1:X.K
        ϕDist = model.K_ϕ(prevtau,prevphi,extendedτ[n],extendedτ[n+1],y,par)
        llk += logpdf(ϕDist,X.ϕ[n])
        prevtau = X.τ[n]
        prevphi = X.ϕ[n]
    end
    return llk
end
function G(X,t1,y,model,par,samplingdensity)
    return log_pdmp_posterior(X,t1,y,model,par) - samplingdensity
end
function G(X,J1,t0,t1,J0,y,model,par,samplingdensity)
    if J1.K == J0.K
        logS = model.logS(J1.τ[end],t1,par) - model.logS(J0.τ[end],t0,par)
        logτ = 0.0
        logϕ = 0.0
        llk  = model.CalLlk(J1,y,t0,t1,par)
    else
        logS = model.logS(J1.τ[end],t1,par) - model.logS(J0.τ[end],t0,par)
        logτ = sum(model.logf.(X.τ,[[J0.τ[end]];X.τ[1:end-1]],Ref(par)))
        logϕ = sum(model.logg.(X.ϕ,[[J0.ϕ[end]];X.ϕ[1:end-1]],[[J0.τ[end]];X.τ[1:end-1]],X.τ,Ref(par)))
        llk = model.CalLlk(J1,y,t0,t1,par)
    end
    return logS + logτ + logϕ + llk - samplingdensity
end
function addPDMP(prevξ,newproc)
    K = prevξ.K + newproc.K
    τ = [prevξ.τ;newproc.τ]
    ϕ = [prevξ.ϕ;newproc.ϕ]
    return PDMP(K,τ,ϕ)
end
function insertPDMP(oldproc,laterξ)
    K = oldproc.K +  laterξ.K
    τ = [oldproc.τ;laterξ.τ]
    ϕ = [oldproc.ϕ;laterξ.ϕ]
    return PDMP(K,τ,ϕ)
end
function SMC(N,T,y,model,par)
    P = length(T)-1
    X = Matrix{Any}(undef,N,P)
    J = Matrix{Any}(undef,N,P)
    logW = zeros(N,P)
    W = zeros(N,P)
    A = zeros(Int64,N,P-1)
    SampDenMat = zeros(N,P)
    ESS = zeros(P)
    for i = 1:N
        X[i,1],SampDenMat[i,1] = KX(T[2],y,model,par)
        while isnan(SampDenMat[i,1]) | isinf(SampDenMat[i,1])
            X[i,1],SampDenMat[i,1] = KX(T[2],y,model,par)
        end
        J[i,1] = X[i,1]
        logW[i,1] = G(X[i,1],T[2],y,model,par,SampDenMat[i,1])
    end
    MAX,_ = findmax(logW[:,1])
    W[:,1] = exp.(logW[:,1] .- MAX); W[:,1] = W[:,1]/sum(W[:,1])
    ESS[1] = 1/sum(W[:,1].^2)
    for n = 2:P
        if ESS[n-1] < 0.5*N
            A[:,n-1] = vcat(fill.(1:N,rand(Multinomial(N,W[:,n-1])))...)
            prevweight = ones(N)/N
        else
            A[:,n-1] = collect(1:N)
            prevweight = W[:,n-1]
        end
        for i = 1:N 
            X[i,n],SampDenMat[i,n] = KX(T[n],T[n+1],J[A[i,n-1],n-1],y,model,par)
            while isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                X[i,n],SampDenMat[i,n] = KX(T[n],T[n+1],J[A[i,n-1],n-1],y,model,par)
            end
            J[i,n] = addPDMP(J[A[i,n-1],n-1],X[i,n])
            logW[i,n] = G(X[i,n],J[i,n],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,SampDenMat[i,n]) + prevweight[i]
        end
        MAX,_ = findmax(logW[:,n])
        W[:,n] = exp.(logW[:,n] .- MAX); W[:,n] = W[:,n]/sum(W[:,n])
        ESS[n] = 1/sum(W[:,n].^2)
    end
    return (X=X,J=J,logW=logW,W=W,A=A,SampDenMat=SampDenMat,ESS=ESS)
end
function BackG(J0,Jlater,y,t0,t1,tP,model,par)
    if Jlater.K == 0
        logS = model.logS(J0.τ[end],tP,par) - model.logS(J0.τ[end],t1,par)
        logτ = 0.0
        logϕ = 0.0
        llk = model.CalLlk(insertPDMP(J0,Jlater),y,t1,tP,par)
    else
        logS = - model.logS(J0.τ[end],t1,par)
        logτ = model.logf(Jlater.τ[1],J0.τ[end],par)
        logϕ = model.logg(Jlater.ϕ[1],J0.ϕ[end],J0.τ[end],Jlater.τ[1],par)
        llk = model.CalLlk(insertPDMP(J0,Jlater),y,t1,Jlater.τ[1],par)
    end
    return logS + logτ + logϕ + llk
end
function BS(R,y,T,model,par)
    N,P = size(R.W)
    BW = zeros(N,P)
    BW[:,P] = R.W[:,P]
    BVec = zeros(Int64,P)
    BVec[P] = vcat(fill.(1:N,rand(Multinomial(1,BW[:,P])))...)[1]
    Jlater = R.X[BVec[P],P]
    L = Vector{Any}(undef,P)
    L[P] = R.X[BVec[P],P]
    for t = (P-1):-1:1
        for i = 1:N
            BW[i,t] = R.logW[i,t] + BackG(R.J[i,t],Jlater,y,T[t],T[t+1],T[end],model,par)
        end
        MAX,_ = findmax(BW[:,t])
        BW[:,t] = exp.(BW[:,t] .- MAX)
        BW[:,t] = BW[:,t]/sum(BW[:,t])
        BVec[t] = vcat(fill.(1:N,rand(Multinomial(1,BW[:,t])))...)[1]
        L[t] = R.X[BVec[t],t]
        Jlater = insertPDMP(L[t],Jlater)
    end
    return (Path=Jlater,X = L,Indices=BVec)
end
function cSMC(L,N,T,y,model,par)
    P = length(T)-1
    X = Matrix{Any}(undef,N,P)
    J = Matrix{Any}(undef,N,P)
    logW = zeros(N,P)
    W = zeros(N,P)
    A = zeros(Int64,N,P-1)
    SampDenMat = zeros(N,P)
    for i = 1:N
        if i == 1
            X[i,1] = L[1]
            SampDenMat[i,1] = qX(X[i,1],T[2],y,model,par)
            J[i,1] = X[i,1]
            if isnan(SampDenMat[i,1]) | isinf(SampDenMat[i,1])
                logW[i,1] = -Inf
            else
                logW[i,1] = G(X[i,1],T[2],y,model,par,SampDenMat[i,1])
            end
        else
            X[i,1],SampDenMat[i,1] = KX(T[2],y,model,par)
            while isnan(SampDenMat[i,1]) | isinf(SampDenMat[i,1])
                X[i,1],SampDenMat[i,1] = KX(T[2],y,model,par)
            end
            J[i,1] = X[i,1]
            logW[i,1] = G(X[i,1],T[2],y,model,par,SampDenMat[i,1])
        end
    end
    MAX,_ = findmax(logW[:,1])
    W[:,1] = exp.(logW[:,1] .- MAX); W[:,1] = W[:,1]/sum(W[:,1])
    for n = 2:P
        A[:,n-1] = vcat(fill.(1:N,rand(Multinomial(N,W[:,n-1])))...)
        A[1,n-1] = 1
        for i = 1:N
            if i == 1
                X[i,n] = L[n]
                J[i,n] = addPDMP(J[A[i,n-1],n-1],X[i,n])
                SampDenMat[i,n] = qX(X[i,n],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par)
                if any(isinf.(logW[i,1:n-1])) | isnan(SampDenMat[i,n]) | isinf(SampDenMat[i,n])
                    logW[i,n] = -Inf
                else
                    logW[i,n] = G(X[i,n],J[i,n],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,SampDenMat[i,n])
                end
            else
                X[i,n],SampDenMat[i,n] = KX(T[n],T[n+1],J[A[i,n-1],n-1],y,model,par)
                while isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                    X[i,n],SampDenMat[i,n] = KX(T[n],T[n+1],J[A[i,n-1],n-1],y,model,par)
                end
                J[i,n] = addPDMP(J[A[i,n-1],n-1],X[i,n])
                logW[i,n] = G(X[i,n],J[i,n],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,SampDenMat[i,n])
            end
        end
        MAX,_ = findmax(logW[:,n])
        W[:,n] = exp.(logW[:,n] .- MAX); W[:,n] = W[:,n]/sum(W[:,n])
    end
    return (X=X,J=J,logW=logW,W=W,A=A,SampDenMat=SampDenMat)
end
@kwdef mutable struct PGargs
    dim::Int64
    λ0::Float64 = 1.0
    Σ0::Matrix{Float64} = 10.0*Matrix{Float64}(LinearAlgebra.I,dim,dim)
    μ0::Vector{Float64} = zeros(dim)
    NAdapt::Int64 = 50000
    NBurn::Int64 = 20000
    NChain::Int64= 50000
    SMCAdaptN::Int64 = 10
    SMCN::Int64 = 100
    T::Vector{Float64}
    NFold::Int64 = 500
    Globalalpha::Float64 = 0.234
    Componentalpha::Float64 = 0.5
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
function LogPosteriorRatio(oldθ,newθ,J,y,tP,model)
    newprior = model.logprior(newθ)
    oldprior = model.logprior(oldθ)
    newllk   = log_pdmp_posterior(J,tP,y,model,model.convert_to_pars(newθ))
    oldllk   = log_pdmp_posterior(J,tP,y,model,model.convert_to_pars(oldθ))
    return newprior+newllk-oldprior-oldllk
end
function TunePars(model,y,T;θ0=nothing,method="Global",kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    if method == "Component"
        λmat = zeros(args.NAdapt+1,model.dim)
        λmat[1,:] = args.λ0 * ones(args.dim)
    elseif method == "Global"
        λvec = zeros(args.NAdapt+1)
        λvec[1] = args.λ0
    else
        throw("Please choose either Global or Component method")
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
    R = SMC(args.SMCAdaptN,args.T,y,model,oldpar)
    BSR = BS(R,y,args.T,model,oldpar)
    Path = BSR.Path
    L = BSR.X
    # update
    @info "Tuning PG parameters..."
    for n = 1:args.NAdapt
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
        println(oldθ,log_pdmp_posterior(Path,args.T[end],y,model,oldpar))
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-10*LinearAlgebra.I
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        R = cSMC(L,args.SMCAdaptN,args.T,y,model,oldpar)
        BSR = BS(R,y,args.T,model,oldpar)
        Path = BSR.Path
        L = BSR.X
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
    llk0 = log_pdmp_posterior(Process,T,y,model,oldpar)
    prior0 = model.logprior(oldθ)
    for n = 1:NIter
        newθ = rand(MultivariateNormal(oldθ,vars))
        prior1 = model.logprior(newθ)
        if prior1 == -Inf
            α = 0.0
        else
            llk1 = log_pdmp_posterior(Process,T,y,model,model.convert_to_pars(newθ))
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
        λ,Σ,θ0 = TunePars(model,y,T;method=method,θ0=θ0,kws...)
    else
        λ,Σ = proppar
    end
    if isnothing(θ0)
        θ[1,:] = rand.(model.prior)
    else
        θ[1,:] = θ0
    end
    oldpar = model.convert_to_pars(θ[1,:])
    R = SMC(args.SMCN,args.T,y,model,oldpar)
    BSR = BS(R,y,args.T,model,oldpar)
    Path = BSR.Path
    L = BSR.X
    @info "Running PG algorithms..."
    @showprogress 2 for n = 1:(args.NBurn+args.NChain)
        θ[n+1,:] = MH(θ[n,:],Path,y,args.NFold,method=method,model=model,T=T[end],λ=λ,Σ=Σ)
        println(θ[n+1,:],log_pdmp_posterior(Path,T[end],y,model,model.convert_to_pars(θ[n+1,:])))
        R = cSMC(L,args.SMCN,args.T,y,model,model.convert_to_pars(θ[n+1,:]))
        BSR = BS(R,y,args.T,model,model.convert_to_pars(θ[n+1,:]))
        Path = BSR.Path
        L = BSR.X
    end
    return θ[(args.NBurn+2):end,:]
end
end