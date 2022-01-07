module BlockVPRF
using Distributions, StatsBase, Random, LinearAlgebra, ProgressMeter
using Base:@kwdef

mutable struct SMCRes
    Particles::Matrix{Any}
    PDMP::Matrix{Any}
    Weights::Matrix{Float64}
    NWeights::Matrix{Float64}
    Ancestor::Matrix{Int64}
end

mutable struct BSRes
    BackwardPath
    L::Vector{Any}
    BackIndex::Vector{Int64}
end


function SMC(N,TimeVec,y;model,par)
    T = length(TimeVec) - 1
    Z = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    for i = 1:N
        Z[i,1], SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
        J[i,1] = Z[i,1]
        W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[:,n-1] = sample(1:N,Weights(NW[:,n-1]),N)
        for i = 1:N
            Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par)
            W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,SampDenMat[i,n])
            J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(Z,J,W,NW,A)
end

function cSMC(L,N,TimeVec,y;model,par)
    T = length(TimeVec)-1
    Z = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    for i = 1:N
        if i == 1
            Z[i,1] = L[1]
            SampDenMat[i,1] = model.CalSampDen(Z[i,1],TimeVec[1],TimeVec[2],y,par)
            J[i,1] = Z[i,1]
            W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        else
            Z[i,1], SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            J[i,1] = Z[i,1]
            W[i,1] = model.JointDensity(J[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        end
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[1,n-1] = 1
        A[2:N,n-1] = sample(1:N,Weights(NW[:,n-1]),N-1)
        for i = 1:N
            if i == 1
                Z[i,n] = L[n]
                SampDenMat[i,n] = model.ProposedZDendity(Z[i,n],J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par)
                W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,SampDenMat[i,n])
                J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
            else
                Z[i,n],SampDenMat[i,n] = model.GenZ(J[A[i,n-1],n-1],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par)
                W[i,n] = model.BlockIncrementalWeight(J[A[i,n-1],n-1],Z[i,n],TimeVec[n-1],TimeVec[n],TimeVec[n+1],y,par,SampDenMat[i,n])
                J[i,n],_ = model.BlockAddPDMP(J[A[i,n-1],n-1],Z[i,n])
            end
            NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
        end
    end
    return SMCRes(Z,J,W,NW,A)
end


function TraceLineage(A,n)
    T = size(A,2)+1
    output = zeros(Int64,T)
    output[T] = n
    for i = (T-1):-1:1
        output[i] = A[output[i+1],i]
    end
    return output
end




end