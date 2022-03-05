module SNC
using Distributions, Random, Plots, Roots,JLD2
using Base:@kwdef
dim = 3
prior = truncated.(Normal.([0.0,0.0,0.0],[sqrt(10),10,sqrt(10)]),[0.0,0.0,0.0],[Inf,Inf,Inf])
logprior(θ) = sum(logpdf.(prior,θ))
@kwdef mutable struct pars
    λτ::Float64 = 1/40
    λϕ::Float64 = 2/3
    κ::Float64  = 1/100
end
mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end
function convert_to_pars(θ)
    return pars(λτ=θ[1],λϕ=θ[2],κ=θ[3])
end
"""
    getζ(t,J,par)
Get the PDMP process values at time t given the jump times and values (J) and the parameters (par)
"""
function getζ(t,J,par)
    index = findlast(J.τ .<= t)
    return J.ϕ[index]*exp(-par.κ*(t-J.τ[index]))
end
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
logf(τ1,τ0,par) = logpdf(Exponential(1/par.λτ),τ1-τ0)
logg(ϕ1,ϕ0,τ0,τ1,par) = logpdf(Exponential(1/par.λϕ),ϕ1 - ϕ0*exp(-par.κ*(τ1-τ0)))
logS(τ,t,par) = logccdf(Exponential(1/par.λτ),t-τ)
logϕ0(ϕ0,par) = logpdf(Exponential(1/par.λϕ),ϕ0)

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

function K_M(J0,t1,par)
    birth_prob = cdf(Exponential(1/par.λτ),t1-J0.τ[end])
    return Binomial(1,birth_prob)
end
function K_τm(M,t0,t1,J0,par,auxpar)
    if M == 1
        return Uniform(max(t0,J0.τ[end]),t1)
    elseif M == 0
        return truncated(Normal(J0.τ[end],auxpar[1]),max(J0.τ[end-1],t0),t1)
    end
end
function μ(t0,t1,J1,par,auxpar)
    index = findlast(J1.τ .< t1)
    if J1.τ[index] <= t0
        return Binomial(1,0)
    else
        return Binomial(1,0.1)
    end     
end
function λτ(t0,t1,J1,par,auxpar)
    index = findlast(J1.τ .< t1)
    return truncated(Normal(J1.τ[index],auxpar[1]),max(t0,J1.τ[index-1]),t1)
end
function λϕ(τ,t0,t1,J1,par,auxpar)
    index = findlast(J1.τ .< t1)
    prevtau = J1.τ[index-1]
    prevphi = J1.ϕ[index-1]
    return truncated(Normal(J1.ϕ[index],auxpar[2]),prevphi*exp(-par.κ*(τ-prevtau)),Inf)
end

end