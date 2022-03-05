module BlockVRPF
using Distributions, Plots, StatsPlots, JLD2, LinearAlgebra, ProgressMeter
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
    end
    return (PDMP(K,[[0.0];τ],ϕ),llk)
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
        ϕ0Dist = model.K_ϕ0(X.τ[1],y,par)
        llk += logpdf(ϕ0Dist,X.ϕ[1])
        extendedτ = [X.τ;[t1]]
        for n = 2:(X.K+1)
            ϕDist = model.K_ϕ(extendedτ[n-1],X.ϕ[n-1],extendedτ[n],extendedτ[n+1],y,par)
            llk += logpdf(ϕDist,X.ϕ[n])
        end
    end
    return llk
end
function KZ(t0,t1,t2,J0,y,model,par,auxpar)
    # Sample the modification variable
    MDist = model.K_M(J0,t1,par)
    M = rand(MDist)
    llk = logpdf(MDist,M)
    if (M == 0) & (J0.τ[end] <= t0)
        τm = nothing
    else
        τmDist = model.K_τm(M,t0,t1,J0,par,auxpar)
        τm = rand(τmDist)
        llk += logpdf(τmDist,τm)
    end
    KDist = model.K_K(t1,t2,y,par)
    K = rand(KDist)
    llk += logpdf(KDist,K)
    τ = sort(rand(Uniform(t1,t2),K))
    llk += sum(log.(collect(1:K)/(t2-t1)))
    # Sample ϕm
    if isnothing(τm)
        ϕm = nothing
    else
        if M == 0
            prevtau = J0.τ[end-1]
            prevphi = J0.ϕ[end-1]
        else
            prevtau = J0.τ[end]
            prevphi = J0.ϕ[end]
        end
        if K == 0
            ϕmDist = model.K_ϕ(prevtau,prevphi,τm,t2,y,par)
        else
            ϕmDist = model.K_ϕ(prevtau,prevphi,τm,τ[1],y,par)
        end
        ϕm = rand(ϕmDist)
        llk += logpdf(ϕmDist,ϕm)
    end
    if isnothing(τm)
        prevtau = J0.τ[end]
        prevphi = J0.ϕ[end]
    else
        prevtau = τm
        prevphi = ϕm 
    end
    extendedτ = [τ;[t2]]
    ϕ = zeros(K)
    for n = 1:K
        ϕDist = model.K_ϕ(prevtau,prevphi,τ[n],extendedτ[n+1],y,par)
        ϕ[n] = rand(ϕDist)
        llk += logpdf(ϕDist,ϕ[n])
        prevtau = τ[n]
        prevphi = ϕ[n]
    end
    return ((M=M,τm=τm,ϕm=ϕm,X=PDMP(K,τ,ϕ)),llk)
end
function qZ(Z,t0,t1,t2,J0,y,model,par,auxpar)
    MDist = model.K_M(J0,t1,par)
    llk = logpdf(MDist,Z.M)
    if !isnothing(Z.τm)
        τmDist = model.K_τm(Z.M,t0,t1,J0,par,auxpar)
        llk += logpdf(τmDist,Z.τm)
    end
    KDist = model.K_K(t1,t2,y,par)
    llk += logpdf(KDist,Z.X.K)
    llk += sum(log.(collect(1:Z.X.K)/(t2-t1)))
    extendedτ = [Z.X.τ;[t2]]
    if !isnothing(Z.ϕm)
        if Z.M == 0
            prevtau = J0.τ[end-1]
            prevphi = J0.ϕ[end-1]
        else
            prevtau = J0.τ[end]
            prevphi = J0.ϕ[end]
        end
        ϕmDist = model.K_ϕ(prevtau,prevphi,Z.τm,extendedτ[1],y,par)
        llk += logpdf(ϕmDist,Z.ϕm)
    end
    if isnothing(Z.τm)
        prevtau = J0.τ[end]
        prevphi = J0.ϕ[end]
    else
        prevtau = Z.τm
        prevphi = Z.ϕm 
    end
    for n = 1:Z.X.K
        ϕDist = model.K_ϕ(prevtau,prevphi,Z.X.τ[n],extendedτ[n+1],y,par)
        llk += logpdf(ϕDist,Z.X.ϕ[n])
        prevtau = Z.X.τ[n]
        prevphi = Z.X.ϕ[n]
    end
    return llk 
end
function AddPDMP(J0,Z)
    if Z.M == 1
        K = J0.K + 1 + Z.X.K 
        tau = [J0.τ;[Z.τm];Z.X.τ]
        phi = [J0.ϕ;[Z.ϕm];Z.X.ϕ]
        ubar = (τ=nothing,ϕ=nothing)
    else
        K = J0.K + Z.X.K 
        if isnothing(Z.τm)
            tau = [J0.τ;Z.X.τ]
            phi = [J0.ϕ;Z.X.ϕ]
            ubar = (τ=nothing,ϕ=nothing)
        else
            tau = [J0.τ[1:end-1];[Z.τm];Z.X.τ]
            phi = [J0.ϕ[1:end-1];[Z.ϕm];Z.X.ϕ]
            ubar = (τ=J0.τ[end],ϕ=J0.ϕ[end])
        end
    end
    return (PDMP(K,tau,phi),ubar)
end  
function G(Z,t0,t1,t2,J0,y,model,par,auxpar,propdensity)
    J1,ubar = AddPDMP(J0,Z)
    logS = model.logS(J1.τ[end],t2,par) - model.logS(J0.τ[end],t1,par)
    if Z.M == 1
        if Z.X.K == 0
            logτ = model.logf(Z.τm,J0.τ[end],par)
            logϕ = model.logg(Z.ϕm,J0.ϕ[end],J0.τ[end],Z.τm,par)
            llk  = model.CalLlk(J1,y,Z.τm,t2,par) - model.CalLlk(J0,y,Z.τm,t1,par)
            logμ = logpdf(model.μ(t0,t1,J1,par,auxpar),Z.M)
            logλ = 0.0
        else
            logτ = sum(model.logf.([[Z.τm];Z.X.τ],[[J0.τ[end],Z.τm];Z.X.τ[1:end-1]],Ref(par)))
            logϕ = sum(model.logg.([[Z.ϕm];Z.X.ϕ],[[J0.ϕ[end],Z.ϕm];Z.X.ϕ[1:end-1]],[[J0.τ[end],Z.τm];Z.X.τ[1:end-1]],[[Z.τm];Z.X.τ],Ref(par)))
            llk = model.CalLlk(J1,y,Z.τm,t2,par) - model.CalLlk(J0,y,Z.τm,t1,par)
            logμ = logpdf(model.μ(t0,t1,J1,par,auxpar),Z.M)
            logλ = 0.0
        end
    else
        if isnothing(Z.τm)
            if Z.X.K == 0
                logτ = 0.0
                logϕ = 0.0
                llk  = model.CalLlk(J1,y,t1,t2,par)
            else
                logτ = sum(model.logf.(Z.X.τ,[[J0.τ[end]];Z.X.τ[1:end-1]],Ref(par)))
                logϕ = sum(model.logg.(Z.X.ϕ,[[J0.ϕ[end]];Z.X.ϕ[1:end-1]],[[J0.τ[end]];Z.X.τ[1:end-1]],Z.X.τ,Ref(par)))
                llk  = model.CalLlk(J1,y,t1,t2,par)
            end
            logμ = logpdf(model.μ(t0,t1,J1,par,auxpar),Z.M)
            logλ = 0.0
        else
            llk = model.CalLlk(J1,y,min(J0.τ[end],Z.τm),t2,par) - model.CalLlk(J0,y,min(J0.τ[end],Z.τm),t1,par)
            if Z.X.K == 0
                logτ = model.logf(Z.τm,J0.τ[end-1],par) - model.logf(J0.τ[end],J0.τ[end-1],par)
                logϕ = model.logg(Z.ϕm,J0.ϕ[end-1],J0.τ[end-1],Z.τm,par) - model.logg(J0.ϕ[end],J0.ϕ[end-1],J0.τ[end-1],J0.τ[end],par)
            else
                logτ = sum(model.logf.([[Z.τm];Z.X.τ],[[J0.τ[end-1],Z.τm];Z.X.τ[1:end-1]],Ref(par))) - model.logf(J0.τ[end],J0.τ[end-1],par)
                logϕ = sum(model.logg.([[Z.ϕm];Z.X.ϕ],[[J0.ϕ[end-1],Z.ϕm];Z.X.ϕ[1:end-1]],[[J0.τ[end-1],Z.τm];Z.X.τ[1:end-1]],[[Z.τm];Z.X.τ],Ref(par))) - model.logg(J0.ϕ[end],J0.ϕ[end-1],J0.τ[end-1],J0.τ[end],par)
            end
            logμ = logpdf(model.μ(t0,t1,J1,par,auxpar),Z.M)
            logλ = logpdf(model.λτ(t0,t1,J1,par,auxpar),ubar.τ)
            logλ += logpdf(model.λϕ(ubar.τ,t0,t1,J1,par,auxpar),ubar.ϕ)
        end
    end
    return logS + logτ + logϕ + llk + logμ + logλ - propdensity
end
function G(X,t1,y,model,par,samplingdensity)
    return log_pdmp_posterior(X,t1,y,model,par) - samplingdensity
end
function SMC(N,T,y,model,par,auxpar)
    P = length(T) - 1
    Z = Matrix{Any}(undef,N,P)
    J = Matrix{Any}(undef,N,P)
    logW = zeros(N,P)
    W = zeros(N,P)
    A = zeros(Int64,N,P-1)
    ESS = zeros(P)
    SampDenMat = zeros(N,P)
    for i = 1:N
        X, SampDenMat[i,1] = KX(T[2],y,model,par)
        counter = 0
        while isinf(SampDenMat[i,1]) | isnan(SampDenMat[i,1])
            X, SampDenMat[i,1] = KX(T[2],y,model,par)
            counter += 1
            if counter > 100
                break
            end
        end
        Z[i,1] = (M = 1,τm = nothing, ϕm = nothing, X = X)
        J[i,1] = Z[i,1].X
        if isinf(SampDenMat[i,1]) | isnan(SampDenMat[i,1])
            lowW[i,1] = -Inf
        else
            logW[i,1] = G(X,T[2],y,model,par,SampDenMat[i,1])
        end
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
            Z[i,n],SampDenMat[i,n] = KZ(T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar)
            counter = 0
            while isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                Z[i,n],SampDenMat[i,n] = KZ(T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar)
                counter += 1
                if counter > 100
                    break
                end
            end
            J[i,n],_ = AddPDMP(J[A[i,n-1],n-1],Z[i,n])
            if isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                logW[i,n] = -Inf
            else
                logW[i,n] = G(Z[i,n],T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar,SampDenMat[i,n]) + prevweight[i]
            end
        end
        MAX,_ = findmax(logW[:,n])
        W[:,n] = exp.(logW[:,n] .- MAX); W[:,n] = W[:,n]/sum(W[:,n])
        ESS[n] = 1/sum(W[:,n].^2)
    end
    return (Z=Z,J=J,logW=logW,A=A,W=W,SampDenMat=SampDenMat,ESS=ESS)
end
function BackG(J0,Zstar,J1,t0,t1,tP,y,model,par,auxpar)
    if Zstar.M == 1
        if J0.τ[end] > Zstar.τm
            return -Inf
        else
            NewJ = PDMP(J0.K+1+J1.K,[J0.τ;[Zstar.τm];J1.τ],[J0.ϕ;[Zstar.ϕm];J1.ϕ])
            logτ = model.logf(Zstar.τm,J0.τ[end],par)
            logϕ = model.logg(Zstar.ϕm,J0.ϕ[end],J0.τ[end],Zstar.τm,par)
            llk = -model.CalLlk(J0,y,Zstar.τm,t1,par)
            logS = -model.logS(J0.τ[end],t1,par)
            logμ = logpdf(model.μ(t0,t1,NewJ,par,auxpar),Zstar.M)
            logλ = 0.0
            return logμ + logλ + logτ + logϕ + llk + logS
        end
    else
        if isnothing(Zstar.τm)
            if J0.τ[end] > t0
                return -Inf
            else
                NewJ = PDMP(J0.K + J1.K,[J0.τ;J1.τ],[J0.ϕ;J1.ϕ])
                logλ = 0.0
                logμ = logpdf(model.μ(t0,t1,NewJ,par,auxpar),Zstar.M)
                if J1.K == 0
                    logτ = 0.0
                    logϕ = 0.0
                    logS = model.logS(J0.τ[end],tP,par) - model.logS(J0.τ[end],t1,par)
                    llk = model.CalLlk(NewJ,y,t1,tP,par)
                else
                    logτ = model.logf(J1.τ[1],J0.τ[end],par)
                    logϕ = model.logg(J1.ϕ[1],J0.ϕ[end],J0.τ[end],J1.τ[1],par)
                    logS = -model.logS(J0.τ[end],t1,par)
                    llk  = model.CalLlk(NewJ,y,t1,J1.τ[1],par)
                end
                return logμ + logλ + logτ + logϕ + logS + llk
            end
        else
            if (J0.τ[end]<=t0) 
                return -Inf
            elseif (J0.τ[end-1]>Zstar.τm)
                return -Inf
            else
                NewJ = PDMP(J0.K + J1.K,[J0.τ[1:end-1];[Zstar.τm];J1.τ],[J0.ϕ[1:end-1];[Zstar.ϕm];J1.ϕ])
                logμ = logpdf(model.μ(t0,t1,NewJ,par,auxpar),Zstar.M)
                logλ = logpdf(model.λτ(t0,t1,NewJ,par,auxpar),J0.τ[end])
                logλ += logpdf(model.λϕ(J0.τ[end],t0,t1,NewJ,par,auxpar),J0.ϕ[end])
                logτ = model.logf(Zstar.τm,J0.τ[end-1],par) - model.logf(J0.τ[end],J0.τ[end-1],par)
                logϕ = model.logg(Zstar.ϕm,J0.ϕ[end-1],J0.τ[end-1],Zstar.τm,par) - model.logg(J0.ϕ[end],J0.ϕ[end-1],J0.τ[end-1],J0.τ[end],par)
                logS = -model.logS(J0.τ[end],t1,par)
                llk = model.CalLlk(NewJ,y,min(J0.τ[end],Zstar.τm),Zstar.τm,par) - model.CalLlk(J0,y,min(J0.τ[end],Zstar.τm),t1,par)
                return logμ + logλ + logτ + logϕ + logS + llk
            end
        end
    end
end
function BS(R,y,T,model,par,auxpar)
    N,P = size(R.W)
    BW = zeros(N,P)
    BW[:,P] = R.W[:,P]
    logBW = zeros(N,P)
    I = zeros(Int64,P)
    L = Vector{Any}(undef,P)
    I[P] = vcat(fill.(1:N,rand(Multinomial(1,R.W[:,P])))...)[1]
    L[P] = R.Z[I[P],P]
    J1 = L[P].X
    for t = (P-1):-1:1
        for i = 1:N
            if R.W[i,t] == 0.0
                logBW[i,t] = -Inf
            else
                logBW[i,t] = R.logW[i,t] + BackG(R.J[i,t],L[t+1],J1,T[t],T[t+1],T[end],y,model,par,auxpar)
                if isnan(logBW[i,t]) | (logBW[i,t] == Inf)
                    logBW[i,t] = -Inf
                end
            end
        end
        BW[:,t] = exp.(logBW[:,t] .- findmax(logBW[:,t])[1]); BW[:,t] = BW[:,t]/sum(BW[:,t])
        if any(isnan.(BW[:,t]))
            @save "error.jld2" BW L J1 par R logBW
        end
        I[t] = vcat(fill.(1:N,rand(Multinomial(1,BW[:,t])))...)[1]
        L[t] = R.Z[I[t],t]
        if L[t+1].M == 1
            J1 = PDMP(L[t].X.K + 1 + J1.K,[L[t].X.τ;[L[t+1].τm];J1.τ],[L[t].X.ϕ;[L[t+1].ϕm];J1.ϕ])
        else
            if isnothing(L[t+1].τm)
                J1 = PDMP(L[t].X.K + J1.K,[L[t].X.τ;J1.τ],[L[t].X.ϕ;J1.ϕ])
            else
                J1 = PDMP(L[t].X.K + J1.K,[L[t].X.τ[1:end-1];[L[t+1].τm];J1.τ],[L[t].X.ϕ[1:end-1];[L[t+1].ϕm];J1.ϕ])
            end
        end
    end
    return (Path=J1,L=L,I=I,BW=BW)
end
function ReSample(J,T,model,par,auxpar)
    P = length(T) - 1
    M = zeros(Int64,P-1)
    τm = Vector{Any}(undef,P-1)
    ϕm = Vector{Any}(undef,P-1)
    for n = 1:(P-1)
        index = findlast(J.τ .< T[n+1])
        M[n] = rand(model.μ(T[n],T[n+1],J,par,auxpar))
        if M[n] == 1
            τm[n] = nothing
            ϕm[n] = nothing
        elseif (M[n] == 0) & (J.τ[index] <= T[n])
            τm[n] = nothing
            ϕm[n] = nothing
        else
            τm[n] = rand(model.λτ(T[n],T[n+1],J,par,auxpar))
            ϕm[n] = rand(model.λϕ(τm[n],T[n],T[n+1],J,par,auxpar))
        end
    end
    return (M=M,τm=τm,ϕm=ϕm)
end
function Rejuvenate(J,T,model,par,auxpar)
    M,τbar,ϕbar = ReSample(J,T,model,par,auxpar)
    P = length(M) + 1
    τm = Vector{Any}(undef,P-1)
    ϕm = Vector{Any}(undef,P-1)
    X  = Vector{Any}(undef,P)
    Indice = findall(T[1] .<= J.τ .< T[2])
    if M[1] == 1
        X[1] = PDMP(length(Indice)-2,J.τ[Indice[1:end-1]],J.ϕ[Indice[1:end-1]])
        τm[1] = J.τ[Indice[end]]
        ϕm[1] = J.ϕ[Indice[end]]
    else
        if isnothing(τbar[1])
            X[1] = PDMP(length(Indice)-1,J.τ[Indice],J.ϕ[Indice])
            τm[1] = nothing
            ϕm[1] = nothing
        else
            X[1] = PDMP(length(Indice)-1,[J.τ[Indice[1:end-1]];[τbar[1]]],[J.ϕ[Indice[1:end-1]];[ϕbar[1]]])
            τm[1] = J.τ[Indice[end]]
            ϕm[1] = J.ϕ[Indice[end]]
        end
    end
    for n = 2:(P-1)
        Indice = findall(T[n] .<= J.τ .< T[n+1])
        if M[n] == 1
            X[n] = PDMP(length(Indice)-1,J.τ[Indice[1:end-1]],J.ϕ[Indice[1:end-1]])
            τm[n] = J.τ[Indice[end]]
            ϕm[n] = J.ϕ[Indice[end]]
        else
            if isnothing(τbar[n])
                X[n] = PDMP(length(Indice),J.τ[Indice],J.ϕ[Indice])
                τm[n] = nothing
                ϕm[n] = nothing
            else
                X[n] = PDMP(length(Indice),[J.τ[Indice[1:end-1]];[τbar[n]]],[J.ϕ[Indice[1:end-1]];[ϕbar[n]]])
                τm[n] = J.τ[Indice[end]]
                ϕm[n] = J.ϕ[Indice[end]]
            end
        end
    end
    Indice = findall(T[P] .<= J.τ .< T[P+1])
    X[P] = PDMP(length(Indice),J.τ[Indice],J.ϕ[Indice])
    Z = Array{Any}(undef,P)
    Z[1] = (M=1,τm = nothing,ϕm=nothing,X=X[1])
    for n = 2:P
        Z[n] = (M=M[n-1],τm=τm[n-1],ϕm=ϕm[n-1],X = X[n])
    end
    return Z
end
function cSMC(L,N,T,y,model,par,auxpar)
    P = length(T) - 1
    Z = Matrix{Any}(undef,N,P)
    J = Matrix{Any}(undef,N,P)
    logW = zeros(N,P)
    W = zeros(N,P)
    A = zeros(Int64,N,P-1)
    ESS = zeros(P)
    SampDenMat = zeros(N,P)
    for i = 1:N
        if i == 1
            X = L[1].X
            SampDenMat[i,1] = qX(X,T[2],y,model,par)
        else
            X, SampDenMat[i,1] = KX(T[2],y,model,par)
            counter = 0
            while isinf(SampDenMat[i,1]) | isnan(SampDenMat[i,1])
                X, SampDenMat[i,1] = KX(T[2],y,model,par)
                counter += 1
                if counter > 100
                    break
                end
            end
        end
        Z[i,1] = (M = 1,τm = nothing, ϕm = nothing, X = X)
        J[i,1] = Z[i,1].X
        logW[i,1] = G(X,T[2],y,model,par,SampDenMat[i,1])
        if isnan(logW[i,1]) | (logW[i,1]==Inf)
            logW[i,1] = -Inf
        end
    end
    MAX,_ = findmax(logW[:,1])
    W[:,1] = exp.(logW[:,1] .- MAX); W[:,1] = W[:,1]/sum(W[:,1])
    ESS[1] = 1/sum(W[:,1].^2)
    for n = 2:P
        if ESS[n-1] < 0.5*N
            A[:,n-1] = vcat(fill.(1:N,rand(Multinomial(N,W[:,n-1])))...)
            A[1,n-1] = 1
            prevweight = ones(N)/N
        else
            A[:,n-1] = collect(1:N)
            prevweight = W[:,n-1]
        end
        for i = 1:N
            if i == 1
                Z[i,n] = L[n]
                SampDenMat[i,n] = qZ(Z[i,n],T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar)
            else
                Z[i,n],SampDenMat[i,n] = KZ(T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar)
                counter = 0
                while isinf(SampDenMat[i,n]) | isnan(SampDenMat[i,n])
                    Z[i,n],SampDenMat[i,n] = KZ(T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar)
                    counter += 1
                    if counter > 100
                        break
                    end
                end
            end
            J[i,n],_ = AddPDMP(J[A[i,n-1],n-1],Z[i,n])
            logW[i,n] = G(Z[i,n],T[n-1],T[n],T[n+1],J[A[i,n-1],n-1],y,model,par,auxpar,SampDenMat[i,n]) + prevweight[i]
            if isnan(logW[i,n]) | (logW[i,n]==Inf)
                logW[i,n] = -Inf
            end
        end
        MAX,_ = findmax(logW[:,n])
        W[:,n] = exp.(logW[:,n] .- MAX); W[:,n] = W[:,n]/sum(W[:,n])
        ESS[n] = 1/sum(W[:,n].^2)
    end
    return (Z=Z,J=J,logW=logW,A=A,W=W,SampDenMat=SampDenMat,ESS=ESS)
end
function LogPosteriorRatio(oldθ,newθ,process,y,End,model)
    newprior = model.logprior(newθ)
    oldprior = model.logprior(oldθ)
    newllk   = log_pdmp_posterior(process,End,y,model,model.convert_to_pars(newθ))
    oldllk   = log_pdmp_posterior(process,End,y,model,model.convert_to_pars(oldθ))
    return newprior+newllk-oldprior-oldllk
end
#------------------------------------- Idea Trials ---------------------------------#
@kwdef mutable struct PGargs
    dim::Int64
    λ0::Float64 = 1.0
    Σ0::Matrix{Float64} = 1.0*Matrix{Float64}(LinearAlgebra.I,dim,dim)
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

function TunePars(model,y,T;θ0=nothing,auxpar,kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    λvec = zeros(args.NAdapt+1)
    λvec[1] = args.λ0
    Σ    = args.Σ0
    μ    = args.μ0
    if isnothing(θ0)
        oldθ = rand.(model.prior)
    else
        oldθ =θ0
    end
    oldpar = model.convert_to_pars(oldθ)
    R = SMC(args.SMCAdaptN,args.T,y,model,oldpar,auxpar)
    BSR = BS(R,y,args.T,model,oldpar,auxpar)
    Path = BSR.Path
    @info "Tuning PG parameters"
    @showprogress 1 for n = 1:args.NAdapt
        Z = rand(MultivariateNormal(zeros(args.dim),λvec[n]*Σ))
        newθ = oldθ.+ Z
        newpar = model.convert_to_pars(newθ)
        if model.logprior(newθ) > -Inf
            LLkRatio = LogPosteriorRatio(oldθ,newθ,Path,y,args.T[end],model)
            α = min(1,exp(LLkRatio))
        else
            α = 0.0
        end
        λvec[n+1] = exp(log(λvec[n])+n^(-1/3)*(α - args.Globalalpha))
        if rand() < α
            oldpar = newpar
            oldθ = newθ
        end
        println(oldθ,Path.K)
        #println(oldθ)
        Σ = Σ + n^(-1/3)*((oldθ.-μ)*transpose(oldθ.-μ)-Σ)+1e-10*I
        μ = μ .+ n^(-1/3)*(oldθ .- μ)
        L = Rejuvenate(Path,T,model,oldpar,auxpar)
        R = cSMC(L,args.SMCAdaptN,args.T,y,model,oldpar,auxpar)
        BSR = BS(R,y,args.T,model,oldpar,auxpar)
        Path = BSR.Path
    end
    return (λvec,Σ,oldθ)
end
function MH(θ0,Process,y,NIter;model,T,λ,Σ)
    # Create Output Arrray
    vars = λ*Σ
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
function PG(model,y,T;proppar=nothing,θ0=nothing,auxpar,kws...)
    args = PGargs(;dim=model.dim,T=T,kws...)
    θ = zeros(args.NBurn+args.NChain+1,args.dim)
    if isnothing(proppar)
        λ,Σ,θ0 = TunePars(model,y,T;θ0=θ0,auxpar=auxpar,kws...)
    else
        λ,Σ = proppar
    end
    if isnothing(θ0)
        θ[1,:] = rand.(model.prior)
    else
        θ[1,:] = θ0
    end
    oldpar = model.convert_to_pars(θ[1,:])
    R = SMC(args.SMCAdaptN,args.T,y,model,oldpar,auxpar)
    BSR = BS(R,y,args.T,model,oldpar,auxpar)
    Path = BSR.Path
    @info "Running PG algorithm with BlockVRPF method.."
    @showprogress 1 for n = 1:(args.NBurn+args.NChain)
        θ[n+1,:] = MH(θ[n,:],Path,y,args.NFold,model=model,T=args.T[end],λ=λ,Σ=Σ)
        println(θ[n+1,:],log_pdmp_posterior(Path,T[end],y,model,model.convert_to_pars(θ[n+1,:])),"  ",Path.K)
        #L = BSR.L
        L = Rejuvenate(Path,args.T,model,model.convert_to_pars(θ[n+1,:]),auxpar)
        R = cSMC(L,args.SMCN,args.T,y,model,model.convert_to_pars(θ[n+1,:]),auxpar)
        BSR = BS(R,y,args.T,model,model.convert_to_pars(θ[n+1,:]),auxpar)
        Path = BSR.Path
    end
    return θ[(args.NBurn+2):end,:]
end
    
end

