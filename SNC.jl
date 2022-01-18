module SNC
using Distributions, Random, Plots
using Base:@kwdef
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


end