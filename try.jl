include("SNC.jl")

J,y = SNC.SimData(seed=17372)
histogram(y,bins=250)
par = SNC.pars()
ζ = SNC.getζ.(t,Ref(J),Ref(par))
plot(t,ζ)
