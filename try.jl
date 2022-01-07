using Distributions
par = ChangePoint.pars()
X,llk = ChangePoint.GenParticle(0.0,100.0,y,par)

J0 = X; t0,t1,t2 = 0.0,100.0,200.0


# Let's try to compute the backward incremental weights
BWeights = 