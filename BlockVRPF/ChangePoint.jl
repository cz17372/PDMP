module ChangePoint

using Distributions,Random,Plots
using Base:@kwdef
theme(:ggplot2)
@kwdef mutable struct pars
    ρ::Float64 = 0.9
    σϕ::Float64 = 1.0
    σy::Float64 = sqrt(0.5)
    α::Float64 = 4.0
    β::Float64 = 10.0
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
function convert_to_pars(θ)
    return pars(ρ=θ[1],σϕ=θ[2],σy=θ[3],α=θ[4],β=θ[5])
end
function Getϕfory(process,time)
    return process.ϕ[findlast(process.τ .< time)]
end
function SimData(;seed=12345,T=1000.0,dt=1,kws...)
    Random.seed!(seed)
    par = pars(;kws...)
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
function ProcObsGraph(proc,y;proclabel="",obslabel="",xlab="Time",ylab="\\xi")
    T = y.t[end]
    scatter(y.t,y.y,label=obslabel,color=:grey,markersize=2.0,markerstrokewidth=0,xlabel=xlab,ylabel=ylab)
    for i = 1:(length(proc.τ)-1)
        plot!(proc.τ[i:(i+1)],repeat([proc.ϕ[i]],2),label="",color=:red,linewidth=2.0)
    end
    plot!([proc.τ[end],T],repeat([proc.ϕ[end]],2),label=proclabel,color=:red,linewidth=2.0)
end
function PlotPDMP(graph,Path,End;color=:green)
    extendedτ = [Path.τ;[1000]]
    for n = 1:length(extendedτ)-1
        plot!(graph,extendedτ[n:n+1],[Path.ϕ[n],Path.ϕ[n]],label="",color=color,linewidth=2.0)
    end
    current()
end
function CalLlk(process,y,Start,End,par)
    # Get the observations that are in the interval [Start,End]
    timeind = findall(Start .< y.t .<= End)
    meanvec = Getϕfory.(Ref(process),y.t[timeind])
    return sum(logpdf.(Normal.(meanvec,par.σy),y.y[timeind]))
end
function PDMPDen(J0,End,par)
    IJ = Gamma(par.α,par.β)
    logτ = sum(logpdf.(IJ,J0.τ[2:end] .- J0.τ[1:end-1]))
    logϕ = sum(logpdf.(Normal.(par.ρ*[[0.0];J0.ϕ[1:end-1]],par.σϕ),J0.ϕ))
    logS = logccdf(IJ,End-J0.τ[end])
    return logτ + logϕ + logS
end
function PDMPDenRatio(J0,End0,J1,End1,par)
    if End0 >= End1
        throw("First PDMP must have smaller ending time the 2nd PDMP")
    end
    IJ = Gamma(par.α,par.β)
    if J1.K-J0.K == 0
        # Means no jump after End0
        logS = logccdf(IJ,End1-J1.τ[end])-logccdf(IJ,End0-J0.τ[end])
        logτ = 0.0
        logϕ = 0.0
    else
        first = length(J0.τ)+1
        logS = logccdf(IJ,End1-J1.τ[end])-logccdf(IJ,End0-J0.τ[end])
        logτ = sum(logpdf.(IJ,J1.τ[first:end] .- J1.τ[first-1:end-1]))
        logϕ = sum(logpdf.(Normal.(par.ρ*J1.ϕ[first-1:end-1],par.σϕ),J1.ϕ[first:end]))
    end
    return logτ + logϕ + logS
end
function GenX(Start,End,y,par)
    if Start != 0.0
        throw("Not generating jumps in the first block, argument xi0 is required")
    end
    llk = 0.0
    MeanJumpTime = par.α * par.β
    # Generate the number of jumps in the block
    K = rand(Poisson((End-Start)/MeanJumpTime))
    # Calculate the log-density of sampling K
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),K)
    # Generate the jumps and calculate the log-densities of sampling those jumps
    τ = [[0.0];sort(rand(Uniform(Start,End),K))]
    llk += sum(log.(collect(1:K)/(End-Start)))
    prevϕ = 0.0
    ϕ = zeros(length(τ))
    # Sampling the corresponding jump values 
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
function GenX(Start,End,J0,y,par)
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
    prevϕ = J0.ϕ[end]
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

end


