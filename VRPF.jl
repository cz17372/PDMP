module VRPF
export SMC
using Distributions, StatsBase, Random
mutable struct SMCRes
    Particles::Matrix{Any}
    PDMP::Matrix{Any}
    Weights::Matrix{Float64}
    NWeights::Matrix{Float64}
    Ancestor::Matrix{Int64}
end
function SMC(N,TimeVec,y;model,par)
    T = length(TimeVec)-1
    X = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    # Sample the particles in the first time block
    for i = 1:N
        X[i,1],SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
        W[i,1] = model.DensityRatio(X[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        J[i,1] = X[i,1]
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[:,n-1] = sample(1:N,Weights(NW[:,n-1]),N)
        for i = 1:N
            X[i,n],SampDenMat[i,n] = model.GenParticle(TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
            W[i,n] = model.DensityRatio(X[i,n],J[A[i,n-1],n-1],y,TimeVec[n],TimeVec[n+1],par) - SampDenMat[i,n]
            J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(X,J,W,NW,A)
end
function cSMC(L,N,TimeVec,y;model,par)
    T = length(TimeVec)-1
    X = Matrix{Any}(undef,N,T)
    J = Matrix{Any}(undef,N,T)
    W = zeros(N,T)
    NW = zeros(N,T)
    A = zeros(Int64,N,T-1)
    SampDenMat = zeros(N,T)
    # Sample the particles in the first time block
    for i = 1:N
        if i == 1
            X[i,1] = L[1]
            SampDenMat[i,1] = model.CalSampDen(X[i,1],TimeVec[1],TimeVec[2],y,par)
            J[i,1] = X[i,1]
            W[i,1] = model.DensityRatio(X[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
        else
            X[i,1],SampDenMat[i,1] = model.GenParticle(TimeVec[1],TimeVec[2],y,par)
            W[i,1] = model.DensityRatio(X[i,1],y,TimeVec[1],TimeVec[2],par) - SampDenMat[i,1]
            J[i,1] = X[i,1]
        end
    end
    NW[:,1] = exp.(W[:,1] .- findmax(W[:,1])[1])/sum(exp.(W[:,1] .- findmax(W[:,1])[1]))
    for n = 2:T
        A[1,n-1] = 1
        A[2:N,n-1] = sample(1:N,Weights(NW[:,n-1]),N-1)
        for i = 1:N
            if i == 1
                X[i,n] = L[n]
                SampDenMat[i,n] = model.CalSampDen(X[i,n],TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
                W[i,n] =  model.DensityRatio(X[i,n],J[A[i,n-1],n-1],y,TimeVec[n],TimeVec[n+1],par) - SampDenMat[i,n]
                J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
            else
                X[i,n],SampDenMat[i,n] = model.GenParticle(TimeVec[n],TimeVec[n+1],J[A[i,n-1],n-1],y,par)
                W[i,n] = model.DensityRatio(X[i,n],J[A[i,n-1],n-1],y,TimeVec[n],TimeVec[n+1],par) - SampDenMat[i,n]
                J[i,n] = model.addPDMP(J[A[i,n-1],n-1],X[i,n])
            end
        end
        NW[:,n] = exp.(W[:,n] .- findmax(W[:,n])[1])/sum(exp.(W[:,n] .- findmax(W[:,n])[1]))
    end
    return SMCRes(X,J,W,NW,A)
end
end