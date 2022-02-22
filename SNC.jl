module SNC
using Distributions, Random, Plots, Roots,JLD2
using Base:@kwdef
dim = 3
@kwdef mutable struct pars
    λτ::Float64 = 1/40
    λϕ::Float64 = 2/3
    κ::Float64  = 1/100
end
function convert_to_pars(θ)
    return pars(λτ=θ[1],λϕ=θ[2],κ=θ[3])
end
logf(τ1,τ0,par) = logpdf(Exponential(1/par.λτ),τ1-τ0)
logg(ϕ1,ϕ0,τ0,τ1,par) = logpdf(Exponential(1/par.λϕ),ϕ1 - ϕ0*exp(-par.κ*(τ1-τ0)))
logS(τ,t,par) = logccdf(Exponential(1/par.λτ),t-τ)
logϕ0(ϕ0,par) = logpdf(Exponential(1/par.λϕ),ϕ0)
function integrateζ(J,t1,t2,par)
    phistart = getζ(t1,J,par)
    phiend   = getζ(t2,J,par)
    index    = findall(t1 .< J.τ .< t2)
    tauvec   = [[t1];J.τ[index];[t2]]
    phivec   = [[phistart];J.ϕ[index];[phiend]]
    res = 0.0
    for n = 1:(length(tauvec)-1)
        res += -1/par.κ*phivec[n]*(exp(-par.κ*(tauvec[n+1]-tauvec[n]))-1.0)
    end
    return res
end
function CalLlk(J,y,Start,End,par)
    index = findall(Start .< y .<= End)
    ζvec  = getζ.(y[index],Ref(J),Ref(par))
    integral = integrateζ(J,Start,End,par)
    return -integral+sum(log.(ζvec))
end
"""
    getζ(t,J,par)
Get the PDMP process values at time t given the jump times and values (J) and the parameters (par)
"""
function getζ(t,J,par)
    index = findlast(J.τ .<= t)
    return J.ϕ[index]*exp(-par.κ*(t-J.τ[index]))
end

function K_ϕ0(EndTime,y,par)
    β = 1/par.κ*(1-exp(-par.κ*(EndTime-0.0))) + par.λϕ
    α = length(findall(0.0 .< y .< EndTime)) + 1
    return truncated(Gamma(α,1/β),0.0,Inf)
end
function K_ϕ(prevtau,prevphi,currenttau,EndTime,y,par)
    β = 1/par.κ*(1-exp(-par.κ*(EndTime-currenttau))) + par.λϕ
    α = length(findall(currenttau .< y .< EndTime)) + 1
    lower_bound = prevphi * exp(-par.κ*(currenttau - prevtau))
    return truncated(Gamma(α,1/β),lower_bound,Inf)
end
function K_K(Start,End,y,par)
    MeanJumpTime = 1/par.λτ
    return Poisson((End-Start)/MeanJumpTime)
end

mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end

function SimData(;seed=12345,T=1000.0,kws...)
    Random.seed!(seed)
    par = pars(;kws...)
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    JumpTimes = [0.0]
    while true
        NewInterJumpTime = rand(IJ)
        if JumpTimes[end]+NewInterJumpTime < T
            push!(JumpTimes,JumpTimes[end]+NewInterJumpTime)
        else
            break
        end
    end
    PhiVec = zeros(length(JumpTimes))
    PhiVec[1] = rand(JV)
    for n = 2:length(PhiVec)
        prevphi = PhiVec[n-1]*exp(-par.κ*(JumpTimes[n]-JumpTimes[n-1]))
        PhiVec[n] = prevphi + rand(JV)
    end
    J = PDMP(length(JumpTimes)-1,JumpTimes,PhiVec)
    maxϕ = findmax(PhiVec)[1]
    y = zeros(0)
    s0 = 0
    while s0 < T
        u = rand()
        w = -log(u)/maxϕ
        s0 += w
        if s0 <= T
            D = rand()
            if D <= getζ(s0,J,par)/maxϕ
                push!(y,s0)
            end
        end
    end
    return (J,y)
end
function GenParticle(Start,End,y,par)
    if Start != 0.0
        throw("Not generating jumps in the first block, argument xi0 is required")
    end
    llk = 0.0
    MeanJumpTime = 1/par.λτ
    # Generate the number of jumps in the block
    K = rand(Poisson((End-Start)/MeanJumpTime))
    # Calculate the log-density of sampling K
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),K)
    # Generate the jumps and calculate the log-densities of sampling those jumps
    τ = [[0.0];sort(rand(Uniform(Start,End),K))]
    llk += sum(log.(collect(1:K)/(End-Start)))
    extendedτ = [τ;[End]]
    ϕ = zeros(length(τ))
    prevtau = 0.0
    prevphi = 0.0
    for i = 1:length(ϕ)
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        ϕ[i] = rand(posterior)
        """
        if isinf(ϕ[i])
            ϕ[i] = gentruncated(posterior)
        end
        """
        llk += logpdf(posterior,ϕ[i])
        prevphi = ϕ[i]
        prevtau = extendedτ[i]
    end
    return (PDMP(K,τ,ϕ),llk)
end
function CalSampDen(X,Start,End,y,par)
    llk = 0.0
    MeanJumpTime = 1/par.λτ
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),X.K)
    llk += sum(log.(collect(1:X.K)/(End-Start)))
    extendedτ = [X.τ;[End]]
    prevphi = 0.0
    prevtau = 0.0
    for i = 1:length(X.ϕ)
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        llk += logpdf(posterior,X.ϕ[i])
        prevphi = X.ϕ[i]
        prevtau = extendedτ[i]
    end
    return llk
end
function GenParticle(Start,End,J0,y,par)
    if Start == 0.0
        @info "Generating jumps in the first block, argument xi0 is discarded..."
        proc,llk = GenParticle(Start,End,y,par)
        return (proc,llk)
    end
    llk = 0.0
    MeanJumpTime = 1/par.λτ
    # Generate the number of jumps in the block
    K = rand(Poisson((End-Start)/MeanJumpTime))
    # Calculate the log-density of sampling K
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),K)
    # Generate the jumps and calculate the log-densities of sampling those jumps
    τ = sort(rand(Uniform(Start,End),K))
    llk += sum(log.(collect(1:K)/(End-Start)))
    extendedτ = [τ;[End]]
    ϕ = zeros(length(τ))
    prevtau = J0.τ[end]
    prevphi = J0.ϕ[end]
    for i = 1:length(ϕ)
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        ϕ[i] = rand(posterior)
        llk += logpdf(posterior,ϕ[i])
        prevphi = ϕ[i]
        prevtau = extendedτ[i]
    end
    return (PDMP(K,τ,ϕ),llk)
end
function CalSampDen(X,Start,End,J0,y,par)
    llk = 0.0
    MeanJumpTime = 1/par.λτ
    # Generate the number of jumps in the block
    K = rand(Poisson((End-Start)/MeanJumpTime))
    # Calculate the log-density of sampling K
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),X.K)
    # Generate the jumps and calculate the log-densities of sampling those jumps
    llk += sum(log.(collect(1:X.K)/(End-Start)))
    extendedτ = [X.τ;[End]]
    prevtau = J0.τ[end]
    prevphi = J0.ϕ[end]
    for i = 1:length(X.ϕ)
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        llk += logpdf(posterior,X.ϕ[i])
        prevphi = X.ϕ[i]
        prevtau = extendedτ[i]
    end
    return llk
end
function addPDMP(prevξ,newproc)
    K = prevξ.K
    tau = prevξ.τ
    phi = prevξ.ϕ
    K += newproc.K 
    tau = [tau;newproc.τ]
    phi = [phi;newproc.ϕ]
    return PDMP(K,tau,phi)
end
function insertPDMP(laterξ,oldproc)
    K = laterξ.K
    tau = laterξ.τ
    phi = laterξ.ϕ
    K += oldproc.K 
    tau = [oldproc.τ;tau]
    phi = [oldproc.ϕ;phi]
    return PDMP(K,tau,phi)
end
function PDMPDen(J0,End,par)
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    lastind = findlast(J0.τ .<= End)
    logτ = sum(logpdf.(IJ,J0.τ[2:lastind] .- J0.τ[1:lastind-1]))
    logϕ = logpdf(JV,J0.ϕ[1])
    for n = 1:(lastind-1)
        logϕ += logpdf(JV,J0.ϕ[n+1]-J0.ϕ[n]*exp(-par.κ*(J0.τ[n+1]-J0.τ[n])))
    end
    logS = logccdf(IJ,End-J0.τ[lastind])
    return logτ+logϕ+logS
end
function JointDensity(J0,y,Start,End,par)
    if Start != 0.0
        throw("Begin time must be 0 if the joint density of the pdmp is calculated")
    end
    logPDMP = PDMPDen(J0,End,par)
    llk  = CalLlk(J0,y,Start,End,par)
    return logPDMP+llk
end
function PDMPDenRatio(J0,End0,J1,End1,par)
    if End0 >= End1
        throw("First PDMP must have smaller ending time the 2nd PDMP")
    end
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    if J1.K-J0.K == 0
        logS = logccdf(IJ,End1-J1.τ[end]) - logccdf(IJ,End0-J0.τ[end])
        logτ = 0.0
        logϕ = 0.0
    else
        first = length(J0.τ)+1
        prevphi = J0.ϕ[end]
        prevtau = J0.τ[end]
        logϕ = 0.0
        logτ = sum(logpdf.(IJ,J1.τ[first:end] .- J1.τ[first-1:end-1]))
        logS = logccdf(IJ,End1-J1.τ[end])-logccdf(IJ,End0-J0.τ[end])
        for n = first:length(J1.ϕ)
            logϕ += logpdf(JV,J1.ϕ[n] - prevphi*exp(-par.κ*(J1.τ[n]-prevtau)))
            prevtau = J1.τ[n]
            prevphi = J1.ϕ[n]
        end
    end
    return logτ+logϕ+logS
end
function JointDensityRatio(J0,End0,J1,End1,y,par)
    PDMPRatio = PDMPDenRatio(J0,End0,J1,End1,par)
    llkratio  = CalLlk(J1,y,End0,End1,par)
    return PDMPRatio+llkratio
end
function BSRatio(prevξ,laterξ,y,Start,End,Final,par)
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    if laterξ.K == 0
        logS = logccdf(IJ,Final-prevξ.τ[end]) - logccdf(IJ,End-prevξ.τ[end])
        logτ = 0.0
        logϕ = 0.0
        llk  = CalLlk(insertPDMP(laterξ,prevξ),y,End,Final,par)
    else
        logS = -logccdf(IJ,End-prevξ.τ[end])
        logτ = logpdf(IJ,laterξ.τ[1]-prevξ.τ[end])
        logϕ = logpdf(JV,laterξ.ϕ[1]-prevξ.ϕ[end]*exp(-par.κ*(laterξ.τ[1]-prevξ.τ[end])))
        llk = CalLlk(insertPDMP(laterξ,prevξ),y,End,laterξ.τ[1],par)
    end
    return logS + logτ + logϕ + llk
end
prior = truncated.(Normal.([0.0,0.0,0.0],[sqrt(10),10,sqrt(10)]),[0.0,0.0,0.0],[Inf,Inf,Inf])
logprior(θ) = sum(logpdf.(prior,θ))
function μ(m,t0,t1,J1)
    tau = J1.τ[findlast(J1.τ .< t1)]
    if tau <= t0
        if m == 1
            return -Inf
        else
            return 0.0
        end
    else
        return log(0.5)
    end
end
function λ(ubar,t0,t1,J1,auxpar,par)
    if isnothing(ubar.tau)
        return 0.0
    else
        ind = findlast(J1.τ .<= t1)
        taudist = truncated(Normal(J1.τ[ind],auxpar[1]) ,max(t0,J1.τ[ind-1]),t1)
        lower_bound = J1.ϕ[ind-1]*exp(-par.κ*(J1.τ[ind]-J1.τ[ind-1]))
        phidist = truncated(Normal(J1.ϕ[ind],auxpar[2]),lower_bound,Inf)
        llk = logpdf(phidist,ubar.phi) + logpdf(taudist,ubar.tau)
        return llk
    end
end 
mutable struct Z
    M::Int64
    taum::Any
    phim::Any
    X::PDMP
end
function GenZ(J0,t0,t1,t2,y,par,auxpar)
    llk = 0.0
    MeanJumpTime = 1/par.λτ
    # Generate the number of jumps in the block
    K = rand(Poisson((t2-t1)/MeanJumpTime))
    # Calculate the log-density of sampling K
    llk += logpdf(Poisson((t2-t1)/MeanJumpTime),K)
    # Generate the jumps and calculate the log-densities of sampling those jumps
    τ = sort(rand(Uniform(t1,t2),K))
    llk += sum(log.(collect(1:K)/(t2-t1)))
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    prob = cdf(IJ,t1-J0.τ[end])
    MProposal = Binomial(1,prob)
    M = rand(MProposal)
    llk += logpdf(MProposal,M)
    if M == 1
        taum = rand(Uniform(max(t0,J0.τ[end]),t1))
        llk += logpdf(Uniform(max(t0,J0.τ[end]),t1),taum)
    else
        if J0.τ[end] <= t0
            taum = nothing
        else
            taum = rand(truncated(Normal(J0.τ[end],auxpar[1]),max(t0,J0.τ[end-1]),t1))
            llk += logpdf(truncated(Normal(J0.τ[end],auxpar[1]),max(t0,J0.τ[end-1]),t1),taum)
        end
    end
    if isnothing(taum)
        phim = nothing
    else
        if K == 0
            β = 1/par.κ*(1-exp(-par.κ*(t2-taum))) + par.λϕ
            α = length(findall(taum .< y .<= t2)) +1
        else
            β = 1/par.κ*(1-exp(-par.κ*(τ[1]-taum))) + par.λϕ
            α = length(findall(taum .< y .<=τ[1])) +1
        end
        if M == 1
            prevphi = J0.ϕ[end]
            prevtau = J0.τ[end]
        else
            prevphi = J0.ϕ[end-1]
            prevtau = J0.τ[end-1]
        end
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(taum-prevtau)),Inf)
        phim = rand(posterior)
        """
        if isinf(phim)
            phim = gentruncated(posterior)
        end
        """
        llk += logpdf(posterior,phim)
    end
    if isnothing(taum)
        prevphi = J0.ϕ[end]
        prevtau = J0.τ[end]
    else
        prevphi = phim
        prevtau = taum
    end
    ϕ = zeros(length(τ))
    extendedτ = [τ;[t2]]
    for i = 1:length(ϕ)
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        ϕ[i] = rand(posterior)
        """
        if isinf(ϕ[i])
            ϕ[i] = gentruncated(posterior)
        end
        """
        llk += logpdf(posterior,ϕ[i])
        prevphi = ϕ[i]
        prevtau = extendedτ[i]
    end
    return (Z(M,taum,phim,PDMP(K,τ,ϕ)),llk)
end
function ProposedZDendity(Z,J0,t0,t1,t2,y,par,auxpar)
    llk = 0.0
    MeanJumpTime = 1/par.λτ
    llk += logpdf(Poisson((t2-t1)/MeanJumpTime),Z.X.K)
    llk += sum(log.(collect(1:Z.X.K)/(t2-t1)))
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    prob = cdf(IJ,t1-J0.τ[end])
    MProposal = Binomial(1,prob)
    llk += logpdf(MProposal,Z.M)
    if Z.M == 1
        # A birth is proposed, proposal a jumpt between J0.τ[end] and 
        llk += logpdf(Uniform(max(t0,J0.τ[end]),t1),Z.taum)
    else
        if J0.τ[end] <= t0
            llk += 0.0
        else
            llk += logpdf(truncated(Normal(J0.τ[end],auxpar[1]),max(t0,J0.τ[end-1]),t1),Z.taum)
        end
    end
    if isnothing(Z.taum)
       llk+=0
    else
        if Z.X.K == 0
            β = 1/par.κ*(1-exp(-par.κ*(t2-Z.taum))) + par.λϕ
            α = length(findall(Z.taum .< y .<= t2)) +1
        else
            β = 1/par.κ*(1-exp(-par.κ*(Z.X.τ[1]-Z.taum))) + par.λϕ
            α = length(findall(Z.taum .< y .<=Z.X.τ[1])) +1
        end
        if Z.M == 1
            prevphi = J0.ϕ[end]
            prevtau = J0.τ[end]
        else
            prevphi = J0.ϕ[end-1]
            prevtau = J0.τ[end-1]
        end
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(Z.taum-prevtau)),Inf)
        llk += logpdf(posterior,Z.phim)
    end
    if isnothing(Z.taum)
        prevphi = J0.ϕ[end]
        prevtau = J0.τ[end]
    else
        prevphi = Z.phim
        prevtau = Z.taum
    end
    extendedτ = [Z.X.τ;[t2]]
    for i = 1:length(Z.X.ϕ)
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        llk += logpdf(posterior,Z.X.ϕ[i])
        prevphi = Z.X.ϕ[i]
        prevtau = extendedτ[i]
    end
    return llk
end
function BlockAddPDMP(J0,Z)
    if Z.M == 1
        K = J0.K + 1 + Z.X.K 
        tau = [J0.τ;[Z.taum];Z.X.τ]
        phi = [J0.ϕ;[Z.phim];Z.X.ϕ]
        ubar = (tau=nothing,phi=nothing)
    else
        K = J0.K + Z.X.K 
        if isnothing(Z.taum)
            tau = [J0.τ;Z.X.τ]
            phi = [J0.ϕ;Z.X.ϕ]
            ubar = (tau=nothing,phi=nothing)
        else
            tau = [J0.τ[1:end-1];[Z.taum];Z.X.τ]
            phi = [J0.ϕ[1:end-1];[Z.phim];Z.X.ϕ]
            ubar = (tau=J0.τ[end],phi=J0.ϕ[end])
        end
    end
    return (PDMP(K,tau,phi),ubar)
end
F(t,prevτ,prevϕ,par) = prevϕ*exp(-par.κ*(t-prevτ))
function BlockIncrementalWeight(J0,Z,t0,t1,t2,y,par,auxpar,propdensity)
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    J1,ubar = BlockAddPDMP(J0,Z)
    logS = logccdf(IJ,t2-J1.τ[end]) - logccdf(IJ,t1-J0.τ[end])
    if Z.M == 1
        if Z.X.K == 0
            logτ = logpdf(IJ,Z.taum - J0.τ[end])
            logϕ = logpdf(JV,Z.phim-F(Z.taum,J0.τ[end],J0.ϕ[end],par))
            llk  = CalLlk(J1,y,Z.taum,t2,par)-CalLlk(J0,y,Z.taum,t1,par)
        else
            logτ = sum(logpdf.(IJ,[[Z.taum];Z.X.τ] .- [[J0.τ[end],Z.taum];Z.X.τ[1:end-1]]))
            logϕ = sum(logpdf.(JV,[[Z.phim];Z.X.ϕ] .- F.([[Z.taum];Z.X.τ],[[J0.τ[end],Z.taum];Z.X.τ[1:end-1]],[[J0.ϕ[end],Z.phim];Z.X.ϕ[1:end-1]],Ref(par))))
            llk  = CalLlk(J1,y,Z.taum,t2,par)-CalLlk(J0,y,Z.taum,t1,par)
        end
    else
        if isnothing(Z.taum)
            if Z.X.K == 0
                logτ = 0.0
                logϕ = 0.0
                llk = CalLlk(J1,y,t1,t2,par)
            else
                logτ = sum(logpdf.(IJ,Z.X.τ .- [[J0.τ[end]];Z.X.τ[1:end-1]]))
                logϕ =  sum(logpdf.(JV,Z.X.ϕ .- F.(Z.X.τ,[[J0.τ[end]];Z.X.τ[1:end-1]],[[J0.ϕ[end]];Z.X.ϕ[1:end-1]],Ref(par))))
                llk = CalLlk(J1,y,t1,t2,par)
            end
        else
            llk = CalLlk(J1,y,min(J0.τ[end],Z.taum),t2,par) - CalLlk(J0,y,min(J0.τ[end],Z.taum),t1,par)
            if Z.X.K == 0
                logτ = logpdf(IJ,Z.taum - J0.τ[end-1])-logpdf(IJ,J0.τ[end]-J0.τ[end-1])
                logϕ = logpdf(JV,Z.phim-F(Z.taum,J0.τ[end-1],J0.ϕ[end-1],par))-logpdf(JV,J0.ϕ[end]-F(J0.τ[end],J0.τ[end-1],J0.ϕ[end-1],par))
            else
                logτ = sum(logpdf.(IJ,[[Z.taum];Z.X.τ] .- [[J0.τ[end-1],Z.taum];Z.X.τ[1:end-1]]))-logpdf(IJ,J0.τ[end]-J0.τ[end-1])
                logϕ = sum(logpdf.(JV,[[Z.phim];Z.X.ϕ] .- F.([[Z.taum];Z.X.τ],[[J0.τ[end-1],Z.taum];Z.X.τ[1:end-1]],[[J0.ϕ[end-1],Z.phim];Z.X.ϕ[1:end-1]],Ref(par)))) - logpdf(JV,J0.ϕ[end]-F(J0.τ[end],J0.τ[end-1],J0.ϕ[end-1],par))
            end
        end
    end
    mu = μ(Z.M,t0,t1,J1)
    lambda   = λ(ubar,t0,t1,J1,auxpar,par)
    """
    println("logS =",logS);println("logτ =",logτ);println("logϕ =",logϕ);println("llk =",llk);println("mu =",mu);println("lambda =",lambda)
    """
    return logS + logτ + logϕ + llk - propdensity + mu + lambda 
end
function Rejuvenate(J,T,auxpar,par)
    P = length(T)-1
    Z0 = Vector{Any}(undef,P)
    X = Vector{Any}(undef,P)
    M = zeros(Int64,P)
    U = Matrix{Any}(undef,P-1,2)
    # Rejuvenate the first time block
    n = 1
    temptau = J.τ[findall(T[n] .<= J.τ .< T[n+1])]
    tempphi = J.ϕ[findall(T[n] .<= J.τ .< T[n+1])]
    if temptau[end] <= T[n]
        M[n] = 0
    else
        M[n] = rand(Binomial(1,0.5))
    end
    if M[n] == 1
        X[n] = PDMP(length(temptau)-2,temptau[1:end-1],tempphi[1:end-1])
        U[n,:] = [temptau[end],tempphi[end]]
    else
        if temptau[end] <= T[n]
            X[n] = PDMP(length(temptau)-1,temptau,tempphi)
            U[n,:] = [nothing,nothing]
        else
            prevtau = J.τ[findlast(J.τ .< temptau[end])]
            prevphi = J.ϕ[findlast(J.τ .< temptau[end])]
            taubar = rand(truncated(Normal(temptau[end],auxpar[1]),max(T[n],prevtau),T[n+1]))
            lower_bound = prevphi * exp(-par.κ*(taubar-prevtau))
            phibar = rand(truncated(Normal(tempphi[end],auxpar[2]),lower_bound,Inf))
            X[n] = PDMP(length(temptau)-1,[temptau[1:end-1];[taubar]],[tempphi[1:end-1];[phibar]])
            U[n,:] = [temptau[end],tempphi[end]]
        end
    end
    for n = 2:P-1
        temptau = J.τ[findall(T[n] .<= J.τ .< T[n+1])]
        tempphi = J.ϕ[findall(T[n] .<= J.τ .< T[n+1])]
        if length(temptau) == 0
            M[n] = 0
        else
            M[n] = rand(Binomial(1,0.5))
        end
        if M[n] == 1
            X[n] = PDMP(length(temptau)-1,temptau[1:end-1],tempphi[1:end-1])
            U[n,:] = [temptau[end],tempphi[end]]
        else
            if length(temptau) == 0
                X[n] = PDMP(length(temptau),temptau,tempphi)
                U[n,:] = [nothing,nothing]
            else
                prevtau = J.τ[findlast(J.τ .< temptau[end])]
                prevphi = J.ϕ[findlast(J.τ .< temptau[end])]
                taubar = rand(truncated(Normal(temptau[end],auxpar[1]),max(T[n],prevtau),T[n+1]))
                lower_bound = prevphi * exp(-par.κ*(taubar-prevtau))
                phibar = rand(truncated(Normal(tempphi[end],auxpar[2]),lower_bound,Inf))
                X[n] = PDMP(length(temptau),[temptau[1:end-1];[taubar]],[tempphi[1:end-1];[phibar]])
                U[n,:] = [temptau[end],tempphi[end]]
            end
        end
    end
    temptau = J.τ[findall(T[P] .<= J.τ .< T[P+1])]
    tempphi = J.ϕ[findall(T[P] .<= J.τ .< T[P+1])]
    X[P] = PDMP(length(temptau),temptau,tempphi)
    Z0[1] = Z(1,nothing,nothing,X[1])
    for n=2:P
        Z0[n] = Z(M[n-1],U[n-1,1],U[n-1,2],X[n])
    end
    return Z0
end
function BlockBSIncrementalWeight(J0,Zstar,J1,y,t0,t1,tP,par,auxpar)
    IJ = Exponential(1/par.λτ)
    JV = Exponential(1/par.λϕ)
    if Zstar.M == 1
        if J0.τ[end] >= Zstar.taum
            return -Inf
        else
            NewJ = PDMP(J0.K+1+J1.K,[J0.τ;[Zstar.taum];J1.τ],[J0.ϕ;[Zstar.phim];J1.ϕ])
            logμ = μ(Zstar.M,t0,t1,NewJ)
            logλ = λ((tau=nothing,phi=nothing),t0,t1,NewJ,auxpar,par)
            logτ = logpdf(IJ,Zstar.taum-J0.τ[end])
            logϕ = logpdf(JV,Zstar.phim - F(Zstar.taum,J0.τ[end],J0.ϕ[end],par))
            llk = -CalLlk(J0,y,Zstar.taum,t1,par)
            logS = -logccdf(IJ,t1-J0.τ[end])
            return logμ + logλ + logτ + logϕ + logS + llk
        end
    else
        if isnothing(Zstar.taum)
            if J0.τ[end] > t0
                return -Inf
            else
                NewJ = PDMP(J0.K + J1.K,[J0.τ;J1.τ],[J0.ϕ;J1.ϕ])
                logμ = μ(Zstar.M,t0,t1,NewJ)
                logλ = λ((tau=nothing,phi=nothing),t0,t1,NewJ,auxpar,par)
                if J1.K == 0
                    logτ = 0.0
                    logϕ = 0.0
                    logS = logccdf(IJ,tP-J0.τ[end])-logccdf(IJ,t1-J0.τ[end])
                    llk  = CalLlk(NewJ,y,t1,tP,par)
                else
                    logτ = logpdf(IJ,J1.τ[1]-J0.τ[end])
                    logϕ = logpdf(JV,J1.ϕ[1] - F(J1.τ[1],J0.τ[end],J0.ϕ[end],par))
                    logS = - logccdf(IJ,t1-J0.τ[end])
                    llk = CalLlk(NewJ,y,t1,J1.τ[1],par)
                end
                return logμ + logλ + logτ + logϕ + logS + llk
            end
        else
            if J0.τ[end] <= t0
                return -Inf
            elseif J0.τ[end-1] > Zstar.taum
                return -Inf
            else
                NewJ = PDMP(J0.K + J1.K,[J0.τ[1:end-1];[Zstar.taum];J1.τ],[J0.ϕ[1:end-1];[Zstar.phim];J1.ϕ])
                logμ = μ(Zstar.M,t0,t1,NewJ)
                logλ = λ((tau=J0.τ[end],phi=J0.ϕ[end]),t0,t1,NewJ,auxpar,par)
                logτ = logpdf(IJ,Zstar.taum-J0.τ[end-1]) - logpdf(IJ,J0.τ[end]-J0.τ[end-1])
                logS = - logccdf(IJ,t1-J0.τ[end])
                logϕ = logpdf(JV,Zstar.phim - F(Zstar.taum,J0.τ[end-1],J0.ϕ[end-1],par)) - logpdf(JV,J0.ϕ[end] - F(J0.τ[end],J0.τ[end-1],J0.ϕ[end-1],par))
                llk  = CalLlk(NewJ,y,min(Zstar.taum,J0.τ[end]),t1,par) - CalLlk(J0,y,min(Zstar.taum,J0.τ[end]),t1,par)
                return logμ + logλ + logτ + logϕ + logS + llk
            end
        end
    end
end
end