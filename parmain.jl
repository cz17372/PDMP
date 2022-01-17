using Distributed, SharedArrays
addprocs(20)
println("Loading data...")
@everywhere using Distributions, Random, StatsBase, LinearAlgebra, JLD2
@everywhere include("ChangePoint.jl")
@everywhere include("VRPF.jl")
@everywhere include("BlockVRPF.jl")
ξ,y = ChangePoint.SimData(seed=2022);
@save "data.jld2" y
@everywhere @load "data.jld2"
VRPF_RES_25par_500fold = SharedArray{Float64}(50000,5,20)
println("Running VRPF with 25 particles, 500 fold")
@sync @distributed for n = 1:20
    VRPF_RES_25par_500fold[:,:,n] = VRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),SMCAdaptN=100,SMCN=25,Globalalpha=0.25)
end
println("Running Block VRPF with 25 particles, 500 fold")
BVRPF_RES_25par_500fold = SharedArray{Float64}(50000,5,20)
@sync @distributed for n = 1:20
    BVRPF_RES_25par_500fold[:,:,n] = BlockVRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),auxpar=[2.0,1.0],SMCAdaptN=100,SMCN=25,Globalalpha=0.25)
end

VRPF_RES_10par_500fold = SharedArray{Float64}(50000,5,20)
println("Running VRPF with 10 particles, 500 fold")
@sync @distributed for n = 1:20
    VRPF_RES_10par_500fold[:,:,n] = VRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),SMCAdaptN=100,SMCN=10,Globalalpha=0.25)
end
println("Running Block VRPF with 10 particles, 500 fold")
BVRPF_RES_10par_500fold = SharedArray{Float64}(50000,5,20)
@sync @distributed for n = 1:20
    BVRPF_RES_10par_500fold[:,:,n] = BlockVRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),auxpar=[2.0,1.0],SMCAdaptN=100,SMCN=10,Globalalpha=0.25)
end

VRPF_RES_50par_500fold = SharedArray{Float64}(50000,5,20)
println("Running VRPF with 50 particles, 500 fold")
@sync @distributed for n = 1:20
    VRPF_RES_50par_500fold[:,:,n] = VRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),SMCAdaptN=100,SMCN=50,Globalalpha=0.25)
end
println("Running Block VRPF with 50 particles, 500 fold")
BVRPF_RES_50par_500fold = SharedArray{Float64}(50000,5,20)
@sync @distributed for n = 1:20
    BVRPF_RES_50par_500fold[:,:,n] = BlockVRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),auxpar=[2.0,1.0],SMCAdaptN=100,SMCN=50,Globalalpha=0.25)
end

VRPF_RES_100par_500fold = SharedArray{Float64}(50000,5,20)
println("Running VRPF with 100 particles, 500 fold")
@sync @distributed for n = 1:20
    VRPF_RES_100par_500fold[:,:,n] = VRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),SMCAdaptN=100,SMCN=100,Globalalpha=0.25)
end
println("Running Block VRPF with 100 particles, 500 fold")
BVRPF_RES_100par_500fold = SharedArray{Float64}(50000,5,20)
@sync @distributed for n = 1:20
    BVRPF_RES_100par_500fold[:,:,n] = BlockVRPF.PG(ChangePoint,y,collect(0.0:100.0:1000.0),auxpar=[2.0,1.0],SMCAdaptN=100,SMCN=100,Globalalpha=0.25)
end