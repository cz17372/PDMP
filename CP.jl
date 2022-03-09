module CP
using Distributions,Random,Plots,Optim
theme(:ggplot2)
using Base:@kwdef
dim = 5
@kwdef mutable struct pars
    ρ::Float64 = 0.9
    σϕ::Float64 = 1.0
    σy::Float64 = sqrt(0.5)
    α::Float64 = 4.0
    β::Float64 = 10.0
end
mutable struct obs
    t::Vector{Float64}
    y::Vector{Float64}
end
mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end
prior = truncated.(Normal.([0.0,0.0,0.0,0.0,0.0],[10,10,sqrt(10),sqrt(10^3),10^2]),[-Inf,0.0,0.0,0.0,0.0],[Inf,Inf,Inf,Inf,Inf])
logprior(θ) = sum(logpdf.(prior,θ))
"""
    convert_to_pars(θ)
convert a vector of parameters θ to the mutable struct `pars` that specifies the model parameters
"""
function convert_to_pars(θ)
    return pars(ρ=θ[1],σϕ=θ[2],σy=θ[3],α=θ[4],β=θ[5])
end
function Getϕfory(process,time)
    return process.ϕ[findlast(process.τ .< time)]
end
# Define functions for calculating the likelihood of the observations
function CalLlk(process,y,Start,End,par)
    # Get the observations that are in the interval [Start,End]
    timeind = findall(Start .< y.t .<= End)
    meanvec = Getϕfory.(Ref(process),y.t[timeind])
    return sum(logpdf.(Normal.(meanvec,par.σy),y.y[timeind]))
end
function SimData(;seed=12345,T=1000.0,dt=1,kws...)
    Random.seed!(seed)
    par = pars(;kws...)
    println(par)
    IJ = Gamma(par.α,par.β)
    ϵϕ = Normal(0,par.σϕ)
    # Generate the jump times
    JumpTimes = [0.0]
    while true
        NewInterJumpTime = rand(IJ)
        if JumpTimes[end] + NewInterJumpTime < T
            push!(JumpTimes,JumpTimes[end]+NewInterJumpTime)
        else
            break
        end
    end
    # Generate the corresponding jump values
    PhiVecs = zeros(length(JumpTimes))
    PhiVecs[1] = rand(ϵϕ)
    for n = 2:length(JumpTimes)
        PhiVecs[n] = par.ρ*PhiVecs[n-1] + rand(ϵϕ)
    end
    Process = PDMP(length(JumpTimes)-1,JumpTimes,PhiVecs)
    timevec = collect(dt:dt:T)
    N = length(timevec)
    y = zeros(N)
    meanvec = Getϕfory.(Ref(Process),timevec)
    y = rand.(Normal.(meanvec,par.σy))
    return (PDMP(length(JumpTimes)-1,JumpTimes,PhiVecs),obs(timevec,y))
end

logf(τ1,τ0,par) = logpdf(Gamma(par.α,par.β),τ1-τ0)
logg(ϕ1,ϕ0,τ0,τ1,par) = logpdf(Normal(0,par.σϕ),ϕ1 - par.ρ*ϕ0)
logϕ0(ϕ0,par) = logpdf(Normal(0,par.σϕ),ϕ0)
logS(τ,t,par) = logccdf(Gamma(par.α,par.β),t-τ)

function K_K(Start,End,y,par)
    MeanJumpTime = par.α * par.β
    return Poisson((End-Start)/MeanJumpTime)
end
function K_ϕ0(EndTime,y,par)
    partialy = y.y[0.0 .< y.t .<= EndTime]
    prevϕ = 0.0
    μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
    sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
    return Normal(μ,sd)
end
function K_ϕ(prevtau,prevphi,currenttau,EndTime,y,par)
    partialy = y.y[currenttau .< y.t .<= EndTime]
    μ = (par.ρ*par.σy^2*prevphi+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
    sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
    return Normal(μ,sd)
end
function K_M(J0,t1,par)
    birth_prob = cdf(Gamma(par.α,par.β),t1-J0.τ[end])
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
        return Binomial(1,0.5)
    end     
end
function λτ(t0,t1,J1,par,auxpar)
    index = findlast(J1.τ .< t1)
    return truncated(Normal(J1.τ[index],auxpar[1]),max(t0,J1.τ[index-1]),t1)
end
function λϕ(τ,t0,t1,J1,par,auxpar)
    index = findlast(J1.τ .< t1)
    return Normal(J1.ϕ[index],auxpar[2])
end
end