module SNC
using Distributions, Random, Plots
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
mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end
"""
    getζ(t,J,par)
Get the PDMP process values at time t given the jump times and values (J) and the parameters (par)
"""
function getζ(t,J,par)
    index = findlast(J.τ .<= t)
    return J.ϕ[index]*exp(-par.κ*(t-J.τ[index]))
end
"""
    integrateζ(J,t1,t2,par)
Calculate the integral of the PDMP defined by J between t1 and t2.
"""
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
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + 1/par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        #posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        posterior = Gamma(α,1/β)
        ϕ[i] = rand(posterior)
        #while isinf(ϕ[i])
        #    ϕ[i] = rand(posterior)
        #end
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
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + 1/par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        #posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        posterior = Gamma(α,1/β)
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
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + 1/par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        #posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        posterior = Gamma(α,1/β)
        ϕ[i] = rand(posterior)
        #while isinf(ϕ[i])
        #    ϕ[i] = rand(posterior)
        #end
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
        β = 1/par.κ*(1-exp(-par.κ*(extendedτ[i+1]-extendedτ[i]))) + 1/par.λϕ
        α = length(findall(extendedτ[i] .< y .<= extendedτ[i+1])) +1
        #posterior = truncated(Gamma(α,1/β),prevphi*exp(-par.κ*(extendedτ[i]-prevtau)),Inf)
        posterior = Gamma(α,1/β)
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
end