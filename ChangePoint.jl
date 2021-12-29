module ChangePoint
using Distributions,Random,Plots
theme(:ggplot2)
using Base:@kwdef
export PDMP
export pars
export obs
dim = 4
@kwdef mutable struct pars
    ρ::Float64 = 0.9
    σϕ::Float64 = 1.0
    σy::Float64 = sqrt(0.5)
    α::Float64 = 4.0
    β::Float64 = 10.0
    θ::Vector{Float64} = [ρ,σϕ,σy,α,β]
end 
mutable struct PDMP
    K::Int64
    τ::Vector{Float64}
    ϕ::Vector{Float64}
end
mutable struct obs
    t::Vector{Float64}
    y::Vector{Float64}
end
function SimData(;seed=12345,T=1000,dt=1,kws...)
    Random.seed!(seed)
    θ = pars(;kws...).θ
    IJ = Gamma(θ[4],θ[5])
    ϵy = Normal(0,θ[3])
    ϵϕ = Normal(0,θ[2])
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
        PhiVecs[n] = θ[1]*PhiVecs[n-1] + rand(ϵy)
    end
    timevec = collect(dt:dt:T)
    N = length(timevec)
    y = zeros(N)
    for n = 1:N
        y[n] = rand(Normal(PhiVecs[findlast(JumpTimes .< timevec[n])],θ[3]))
    end
    return (PDMP(length(JumpTimes),JumpTimes,PhiVecs),obs(timevec,y))
end
function ProcObsGraph(proc,y;proclabel="",obslabel="",xlab="Time",ylab="\\xi")
    T = y.t[end]
    scatter(y.t,y.y,label=obslabel,color=:grey,markersize=2.0,markerstrokewidth=0,xlabel=xlab,ylabel=ylab)
    for i = 1:(length(proc.τ)-1)
        plot!(proc.τ[i:(i+1)],repeat([proc.ϕ[i]],2),label="",color=:red,linewidth=2.0)
    end
    plot!([proc.τ[end],T],repeat([proc.ϕ[end]],2),label=proclabel,color=:red,linewidth=2.0)
end
# Generate jump times and values in time blocks for the SMC part
function GenParticle(Start,End,y,par)
    if Start != 0.0
        throw("Not generating jumps in the first block, argument xi0 is required")
    end
    llk = 0.0
    MeanJumpTime = par.α * par.β
    K = rand(Poisson((End-Start)/MeanJumpTime))
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),K)
    # First generate the jump times in the block
    τ = [[0.0];sort(rand(Uniform(Start,End),K))]
    llk += sum(log.(collect(1:K)/(End-Start)))
    prevϕ = 0.0
    ϕ = zeros(length(τ))
    extendedτ = [τ;[End]]
    for i = 1:length(ϕ)
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(par.σϕ^2 + par.σy^2)
        ϕ[i] = rand(Normal(μ,sd))
        llk += logpdf(Normal(μ,sd),ϕ[i])
        prevϕ = ϕ[i]
    end
    return (PDMP(K,τ,ϕ),llk)
end
function GenParticle(Start,End,prevξ,y,par)
    if Start == 0.0
        @info "Generating jumps in the first block, argument xi0 is discarded..."
        proc,llk = GenParticle(Start,End,y,par)
        return (proc,llk)
    end
    llk = 0.0
    MeanJumpTime = par.α * par.β
    K = rand(Poisson((End-Start)/MeanJumpTime))
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),K)
    # First generate the jump times in the block
    τ = sort(rand(Uniform(Start,End),K))
    llk += sum(log.(collect(1:K)/(End-Start)))
    prevϕ = prevξ.ϕ[end]
    ϕ = zeros(length(τ))
    extendedτ = [τ;[End]]
    for i = 1:length(ϕ)
        # Find the observations that 
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(par.σϕ^2 + par.σy^2)
        ϕ[i] = rand(Normal(μ,sd))
        llk += logpdf(Normal(μ,sd),ϕ[i])
        prevϕ = ϕ[i]
    end
    return (PDMP(K,τ,ϕ),llk)
end
function CalSampDen(X,Start,End,y,par)
    llk = 0.0
    MeanJumpTime = par.α * par.β
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),X.K)
    llk += sum(log.(collect(1:X.K)/(End-Start)))
    extendedτ = [X.τ;[End]]
    prevϕ = 0.0
    for i = 1:length(X.ϕ)
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(par.σϕ^2 + par.σy^2)
        llk += logpdf(Normal(μ,sd),X.ϕ[i])
        prevϕ = X.ϕ[i]
    end
    return llk
end
function CalSampDen(X,Start,End,prevξ,y,par)
    llk = 0.0
    MeanJumpTime = par.α * par.β
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),X.K)
    llk += sum(log.(collect(1:X.K)/(End-Start)))
    prevϕ = prevξ.ϕ[end]
    extendedτ = [X.τ;[End]]
    for i = 1:length(X.ϕ)
        # Find the observations that 
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(par.σϕ^2 + par.σy^2)
        llk += logpdf(Normal(μ,sd),X.ϕ[i])
        prevϕ = X.ϕ[i]
    end
    return llk
end
# Define functions for calculating the joint density of the jump times and values
function CalPDMP(proc,End,par)
    IJ = Gamma(par.α,par.β)
    logτ = sum(logpdf.(IJ,proc.τ[2:end] .- proc.τ[1:end-1]))
    logϕ = sum(logpdf.(Normal.(par.ρ*[[0.0];proc.ϕ[1:end-1]],par.σϕ),proc.ϕ))
    logS = logccdf(IJ,End-proc.τ[end])
    return logτ + logϕ + logS
end
function CalPDMP(proc,Start,End,prevξ,par)
    IJ = Gamma(par.α,par.β)
    if proc.K == 0
        logS = logccdf(IJ,End-prevξ.τ[end])
        logτ = 0.0
        logϕ = 0.0
    else
        logS = logccdf(IJ,End-proc.τ[end])
        logτ = sum(logpdf.(IJ,proc.τ .- [[prevξ.τ[end]];proc.τ[1:end-1]]))
        logϕ = sum(logpdf.(Normal.(par.ρ*[[prevξ.ϕ[end]];proc.ϕ[1:end-1]],par.σϕ),proc.ϕ))
    end
    return logτ + logϕ + logS
end
# Define functions for calculating the likelihood of the observations
function Getϕfory(process,time)
    return process.ϕ[findlast(process.τ .< time)]
end
function CalLlk(process,y,Start,End,par)
    # Get the observations that are in the interval [Start,End]
    timeind = findall(Start .< y.t .<= End)
    meanvec = Getϕfory.(Ref(process),y.t[timeind])
    return sum(logpdf.(Normal.(meanvec,par.σy),y.y[timeind]))
end
function addPDMP(prevξ,newproc...)
    K = prevξ.K
    tau = prevξ.τ
    phi = prevξ.ϕ
    for particle in newproc
        K += particle.K
        tau = [tau;particle.τ]
        phi = [phi;particle.ϕ]
    end
    return PDMP(K,tau,phi)
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
# Define functions used to calculate incremental weights
function DensityRatio(X,y,Start,End,par)
    logPDMP = CalPDMP(X,End,par)
    llk  = CalLlk(X,y,Start,End,par)
    return logPDMP+llk
end
function DensityRatio(X,prevξ,y,Start,End,par)
    logPDMP = CalPDMP(X,Start,End,prevξ,par)
    llk     = CalLlk(addPDMP(prevξ,X),y,Start,End,par)
    logS    = logccdf(Gamma(par.α,par.β),Start-prevξ.τ[end])
    return logPDMP + llk - logS
end
function BSRatio(prevξ,laterξ,y,Start,End,Final,par)
    IJ = Gamma(par.α,par.β)
    if laterξ.K == 0
        logS = logccdf(IJ,Final-prevξ.τ[end]) - logccdf(IJ,Start-prevξ.τ[end])
        logτ = 0.0
        logϕ = 0.0
        llk  = CalLlk(insertPDMP(laterξ,prevξ),y,Start,Final,par)
    else
        logS = -logccdf(IJ,Start-prevξ.τ[end])
        logτ = logpdf(IJ,laterξ.τ[1]-prevξ.τ[end])
        logϕ = logpdf(Normal(par.ρ*prevξ.ϕ[end],par.σϕ),laterξ.ϕ[1])
        llk = CalLlk(insertPDMP(laterξ,prevξ),y,Start,laterξ.τ[1],par)
    end
    return logS+logτ+logϕ+llk
end
prior = truncated.(Normal.([0.0,0.0,0.0,0.0,0.0],[10,10,sqrt(10),sqrt(10^3),10^2]),[-Inf,0.0,0.0,0.0,0.0],[Inf,Inf,Inf,Inf,Inf])
end