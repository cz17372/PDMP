module ChangePoint
using Distributions,Random,Plots
theme(:ggplot2)
using Base:@kwdef
export PDMP
export pars
export obs
dim = 5
@kwdef mutable struct pars
    ρ::Float64 = 0.9
    σϕ::Float64 = 1.0
    σy::Float64 = 0.75
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
function ProcObsGraph(proc,y;kws...)
    T = y.t[end]
    scatter(y.t,y.y,color=:grey,markersize=2.0,markerstrokewidth=0;kws...)
    for i = 1:(length(proc.τ)-1)
        plot!(proc.τ[i:(i+1)],repeat([proc.ϕ[i]],2),label="",color=:red,linewidth=2.0)
    end
    plot!([proc.τ[end],T],repeat([proc.ϕ[end]],2),label="",color=:red,linewidth=2.0)
end
function PlotPDMP(graph,Path,End;color=:green)
    extendedτ = [Path.τ;[1000]]
    for n = 1:length(extendedτ)-1
        plot!(graph,extendedτ[n:n+1],[Path.ϕ[n],Path.ϕ[n]],label="",color=color,linewidth=2.0)
    end
    current()
end
"""
    GenParticle(Start,End,y,par)
Generate jump times and values in the first time block [Start,End]. `Start` must be 0.0 since this is the function used to sample particles in the first time block.
"""
function GenParticle(Start,End,y,par)
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
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        ϕ[i] = rand(Normal(μ,sd))
        llk += logpdf(Normal(μ,sd),ϕ[i])
        prevϕ = ϕ[i]
    end
    return (PDMP(K,τ,ϕ),llk)
end
function GenParticle(Start,End,J0,y,par)
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
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
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
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        llk += logpdf(Normal(μ,sd),X.ϕ[i])
        prevϕ = X.ϕ[i]
    end
    return llk
end
function CalSampDen(X,Start,End,J0,y,par)
    llk = 0.0
    MeanJumpTime = par.α * par.β
    llk += logpdf(Poisson((End-Start)/MeanJumpTime),X.K)
    llk += sum(log.(collect(1:X.K)/(End-Start)))
    prevϕ = J0.ϕ[end]
    extendedτ = [X.τ;[End]]
    for i = 1:length(X.ϕ)
        # Find the observations that 
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        llk += logpdf(Normal(μ,sd),X.ϕ[i])
        prevϕ = X.ϕ[i]
    end
    return llk
end
# Define functions for calculating the joint density of the jump times and values
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
# Define functions for calculating the likelihood of the observations
function CalLlk(process,y,Start,End,par)
    # Get the observations that are in the interval [Start,End]
    timeind = findall(Start .< y.t .<= End)
    meanvec = Getϕfory.(Ref(process),y.t[timeind])
    return sum(logpdf.(Normal.(meanvec,par.σy),y.y[timeind]))
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
function JointDensity(J0,y,Start,End,par)
    if Start != 0.0
        throw("Begin time must be 0 if the joint density of the pdmp is calculated")
    end
    logPDMP = PDMPDen(J0,End,par)
    llk  = CalLlk(J0,y,Start,End,par)
    return logPDMP+llk
end
function JointDensityRatio(J0,End0,J1,End1,y,par)
    PDMPRatio = PDMPDenRatio(J0,End0,J1,End1,par)
    llkratio  = CalLlk(J1,y,End0,End1,par)
    return PDMPRatio+llkratio
end
function BSRatio(prevξ,laterξ,y,Start,End,Final,par)
    IJ = Gamma(par.α,par.β)
    if laterξ.K == 0
        logS = logccdf(IJ,Final-prevξ.τ[end]) - logccdf(IJ,End-prevξ.τ[end])
        logτ = 0.0
        logϕ = 0.0
        llk  = CalLlk(insertPDMP(laterξ,prevξ),y,End,Final,par)
    else
        logS = -logccdf(IJ,End-prevξ.τ[end])
        logτ = logpdf(IJ,laterξ.τ[1]-prevξ.τ[end])
        logϕ = logpdf(Normal(par.ρ*prevξ.ϕ[end],par.σϕ),laterξ.ϕ[1])
        llk = CalLlk(insertPDMP(laterξ,prevξ),y,End,laterξ.τ[1],par)
    end
    return logS+logτ+logϕ+llk
end
prior = truncated.(Normal.([0.0,0.0,0.0,0.0,0.0],[10,10,sqrt(10),sqrt(10^3),10^2]),[-Inf,0.0,0.0,0.0,0.0],[Inf,Inf,Inf,Inf,Inf])
logprior(θ) = sum(logpdf.(prior,θ))
# Block SMC functions...
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
function λ(ubar,t0,t1,J1,auxpar)
    if isnothing(ubar.tau)
        return 0.0
    else
        ind = findlast(J1.τ .<= t1)
        llk = logpdf(Normal(J1.ϕ[ind],auxpar[2]),ubar.phi) + logpdf(truncated(Normal(J1.τ[ind],auxpar[1]) ,max(t0,J1.τ[ind-1]),t1),ubar.tau)
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
    # Generate jump times in [t1,t2] 
    MeanJumpTime = (t2-t1)/(par.α*par.β)
    K = rand(Poisson(MeanJumpTime))
    llk += logpdf(Poisson(MeanJumpTime),K)
    τ = sort(rand(Uniform(t1,t2),K))
    llk += sum(log.(collect(1:K)/(t2-t1)))
    # Generate M 
    IJ = Gamma(par.α,par.β)
    prob = cdf(IJ,t1-J0.τ[end])
    MProposal= Binomial(1,prob)
    M = rand(MProposal)
    llk += logpdf(MProposal,M)
    if M == 1
        # A birth is proposed, proposal a jumpt between J0.τ[end] and 
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
    # Generate ϕm
    if isnothing(taum)
        phim = nothing
    else
        if K == 0
            # No jumps in the block [t1,t2]
            partialy = y.y[taum .< y.t .<= t2]
        else
            partialy = y.y[taum .< y.t .<= τ[1]]
        end
        if M == 1
            prevϕ = J0.ϕ[end]
        else
            prevϕ = J0.ϕ[end-1]
        end
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        phim = rand(Normal(μ,sd))
        llk += logpdf(Normal(μ,sd),phim)
    end
    if isnothing(taum)
        prevϕ=J0.ϕ[end]
    else
        prevϕ = phim
    end
    ϕ = zeros(length(τ))
    extendedτ = [τ;[t2]]
    for i = 1:length(ϕ)
        # Find the observations that 
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        ϕ[i] = rand(Normal(μ,sd))
        llk += logpdf(Normal(μ,sd),ϕ[i])
        prevϕ = ϕ[i]
    end
    return (Z(M,taum,phim,ChangePoint.PDMP(K,τ,ϕ)),llk)
end
function ProposedZDendity(Z,J0,t0,t1,t2,y,par,auxpar)
    llk = 0.0
    MeanJumpTime = (t2-t1)/(par.α*par.β)
    llk += logpdf(Poisson(MeanJumpTime),Z.X.K)
    llk += sum(log.(collect(1:Z.X.K)/(t2-t1)))
    IJ = Gamma(par.α,par.β)
    prob = cdf(IJ,t1-J0.τ[end])
    MProposal= Binomial(1,prob)
    llk += logpdf(MProposal,Z.M)
    if Z.M == 1
        # A birth is proposed, proposal a jumpt between J0.τ[end] and 
        llk += logpdf(Uniform(max(t0,J0.τ[end]),t1),Z.taum)
    else
        if J0.τ[end] <= t0
            llk += 0
        else
            llk += logpdf(truncated(Normal(J0.τ[end],auxpar[1]),max(t0,J0.τ[end-1]),t1),Z.taum)
        end
    end
    if isnothing(Z.taum)
        llk+= 0.0
    else
        if Z.X.K == 0
            # No jumps in the block [t1,t2]
            partialy = y.y[Z.taum .< y.t .<= t2]
        else
            partialy = y.y[Z.taum .< y.t .<= Z.X.τ[1]]
        end
        if Z.M == 1
            prevϕ = J0.ϕ[end]
        else
            prevϕ = J0.ϕ[end-1]
        end
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        llk += logpdf(Normal(μ,sd),Z.phim)
    end
    if isnothing(Z.taum)
        prevϕ=J0.ϕ[end]
    else
        prevϕ = Z.phim
    end
    extendedτ = [Z.X.τ;[t2]]
    for i = 1:length(Z.X.ϕ)
        # Find the observations that 
        partialy = y.y[extendedτ[i] .< y.t .<= extendedτ[i+1]]
        μ = (par.ρ*par.σy^2*prevϕ+par.σϕ^2*sum(partialy))/(par.σy^2 + length(partialy)*par.σϕ^2)
        sd = par.σϕ*par.σy/sqrt(length(partialy)*par.σϕ^2 + par.σy^2)
        llk += logpdf(Normal(μ,sd),Z.X.ϕ[i])
        prevϕ = Z.X.ϕ[i]
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
function BlockIncrementalWeight(J0,Z,t0,t1,t2,y,par,auxpar,propdensity)
    IJ = Gamma(par.α,par.β)
    # Calculate the density ratio related to τ's 
    J1,ubar = BlockAddPDMP(J0,Z)
    logS = logccdf(IJ,t2-J1.τ[end]) - logccdf(IJ,t1-J0.τ[end])
    if Z.M == 1
        if Z.X.K == 0
            logτ = logpdf(IJ,Z.taum-J0.τ[end])
            logϕ = logpdf(Normal(par.ρ*J0.ϕ[end],par.σϕ),Z.phim)
            llk  = CalLlk(J1,y,Z.taum,t2,par)-CalLlk(J0,y,Z.taum,t1,par)
        else
            logτ = sum(logpdf.(IJ,[[Z.taum];Z.X.τ] .- [[J0.τ[end],Z.taum];Z.X.τ[1:end-1]]))
            logϕ = sum(logpdf.(Normal.(par.ρ*[[J0.ϕ[end],Z.phim];Z.X.ϕ[1:end-1]],par.σϕ),[[Z.phim];Z.X.ϕ]))
            llk  = CalLlk(J1,y,Z.taum,t2,par)-CalLlk(J0,y,Z.taum,t1,par)
        end
    else
        if isnothing(Z.taum)
            if Z.X.K == 0
                logτ = 0.0
                logϕ = 0.0
                llk  = CalLlk(J1,y,t1,t2,par)
            else
                logτ = sum(logpdf.(IJ,Z.X.τ .- [[J0.τ[end]];Z.X.τ[1:end-1]]))
                logϕ = sum(logpdf.(Normal.(par.ρ*[[J0.τ[end]];Z.X.τ[1:end-1]]),Z.X.ϕ))
                llk = CalLlk(J1,y,t1,t2,par)
            end
        else
            llk = CalLlk(J1,y,min(J0.τ[end],Z.taum),t2,par) - CalLlk(J0,y,min(J0.τ[end],Z.taum),t1,par)
            if Z.X.K == 0
                logτ = logpdf(IJ,Z.taum - J0.τ[end-1]) - logpdf(IJ,J0.τ[end]-J0.τ[end-1])
                logϕ = logpdf(Normal(par.ρ*J0.ϕ[end-1],par.σϕ),Z.phim) - logpdf(Normal(par.ρ*J0.ϕ[end-1],par.σϕ),J0.ϕ[end])
            else
                logτ = sum(logpdf.(IJ,[[Z.taum];Z.X.τ] .- [[J0.τ[end-1],Z.taum];Z.X.τ[1:end-1]])) - logpdf(IJ,J0.τ[end]-J0.τ[end-1])
                logϕ = sum(logpdf.(Normal.(par.ρ*[[J0.ϕ[end-1],Z.phim];Z.X.ϕ[1:end-1]],par.σϕ),[[Z.phim];Z.X.ϕ])) - logpdf(Normal(par.ρ*J0.ϕ[end-1],par.σϕ),J0.ϕ[end])
            end
        end
    end
    mu = μ(Z.M,t0,t1,J1)
    lambda   = λ(ubar,t0,t1,J1,auxpar)
    return logS + logτ + logϕ + llk - propdensity + mu + lambda
end
function Rejuvenate(J,T,auxpar)
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
            phibar = rand(Normal(tempphi[end],auxpar[2]))
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
                phibar = rand(Normal(tempphi[end],auxpar[2]))
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
    IJ = Gamma(par.α,par.β)
    if Zstar.M == 1
        if J0.τ[end] >= Zstar.taum
            return -Inf
        else
            NewJ = PDMP(J0.K+1+J1.K,[J0.τ;[Zstar.taum];J1.τ],[J0.ϕ;[Zstar.phim];J1.ϕ])
            logμ = μ(Zstar.M,t0,t1,NewJ)
            logλ = λ((tau=nothing,phi=nothing),t0,t1,NewJ,auxpar)
            logτ = logpdf(IJ,Zstar.taum-J0.τ[end])
            logϕ = logpdf(Normal(par.ρ*J0.ϕ[end],par.σϕ),Zstar.phim)
            llk  = -CalLlk(J0,y,Zstar.taum,t1,par)
            logS = -logccdf(IJ,t1-J0.τ[end])
            return logμ + logλ + logτ + logϕ + logS + llk
        end
    else
        # An adjust move is proposed
        if isnothing(Zstar.taum)
            # Means that no jumps in the previous block
            if J0.τ[end] > t0
                return -Inf
            else
                NewJ = PDMP(J0.K + J1.K,[J0.τ;J1.τ],[J0.ϕ;J1.ϕ])
                logμ = μ(Zstar.M,t0,t1,NewJ)
                logλ = λ((tau=nothing,phi=nothing),t0,t1,NewJ,auxpar)
                if J1.K == 0
                    logτ = 0.0
                    logϕ = 0.0
                    logS = logccdf(IJ,tP-J0.τ[end])-logccdf(IJ,t1-J0.τ[end])
                    llk  = CalLlk(NewJ,y,t1,tP,par)
                else
                    logτ = logpdf(IJ,J1.τ[1]-J0.τ[end])
                    logϕ = logpdf(Normal(par.ρ*J0.ϕ[end],par.σϕ),J1.ϕ[1])
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
                logλ = λ((tau=J0.τ[end],phi=J0.ϕ[end]),t0,t1,NewJ,auxpar)
                logτ = logpdf(IJ,Zstar.taum-J0.τ[end-1]) - logpdf(IJ,J0.τ[end]-J0.τ[end-1])
                logS = - logccdf(IJ,t1-J0.τ[end])
                logϕ = logpdf(Normal(par.ρ*J0.ϕ[end-1],par.σϕ),Zstar.phim) - logpdf(Normal(par.ρ*J0.ϕ[end-1],par.σϕ),J0.ϕ[end])
                llk  = CalLlk(NewJ,y,min(Zstar.taum,J0.τ[end]),t1,par) - CalLlk(J0,y,min(Zstar.taum,J0.τ[end]),t1,par)
                return logμ + logλ + logτ + logϕ + logS + llk
            end
        end
    end
end
end