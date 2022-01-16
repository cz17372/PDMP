using Distributed, SharedArrays
addprocs(20)
@everywhere using Distributions, Plots, StatsPlots,Random, StatsBase, LinearAlgebra, JLD2
@everywhere include("ChangePoint.jl")
@everywhere include("VRPF.jl")
@everywhere include("BlockVRPF.jl")
ξ,y = ChangePoint.SimData(seed=2022);
@save "data.jld2" y
@everywhere @load "data.jld2"
VRPF_RES = SharedArray{Float64}(50000,5,20)

@distributed for n = 1:20
    VRPF_RES[:,:,n] = VRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),SMCAdaptN=100,SMCN=25,Globalalpha=0.25)
end

figure = plot(size=(600,600))
for i = 1:20
    density!(figure,VRPF_RES[:,1,i],color=:grey,label="")
end
current()


BVRPF_RES = SharedArray{Float64}(50000,5,20)

@distributed for n = 1:20
    BVRPF_RES[:,:,n] = BlockVRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),auxpar=[2.0,1.0],SMCAdaptN=100,SMCN=25,Globalalpha=0.25)
end