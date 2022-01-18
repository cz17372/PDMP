include("SNC.jl")
include("VRPF.jl")
using Distributions, Plots, StatsPlots, Random, StatsBase
J,y = SNC.SimData(seed=17372)
T = collect(0.0:100.0:1000.0);par=SNC.pars();t = collect(0.0:0.1:1000.0)
R = VRPF.SMC(10000,T,y,model=SNC,par=par)
Index = findmax(R.NWeights[:,end])[2]
J1 = R.PDMP[Index,end]
ζ1 = SNC.getζ.(t,Ref(J1),Ref(par))
ζ0 = SNC.getζ.(t,Ref(J),Ref(par))
BSR = VRPF.BS(R,y,T,model=SNC,par=par)
J2 = BSR.BackwardPath
ζ2 = SNC.getζ.(t,Ref(J2),Ref(par))



θ0 = rand.(SNC.prior)./[100.0,1.0,100.0]
R = VRPF.PG(SNC,y,T,θ0=θ0,SMCN=5,SMCAdaptN=100,Globalalpha=0.25)