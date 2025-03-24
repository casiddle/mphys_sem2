#!/usr/bin/env python

"""
   Create an electron witness beam for qv3d

   Written by John.
"""

import numpy as np
import sys

L=float(sys.argv[1]) #bunch length - um (AWAKE: 60)
Q=float(sys.argv[2]) #bunch charge - pC (AWAKE: 120)
D=float(sys.argv[3]) #plasma density - 1/ccm (AWAKE: 7e14)
R=float(sys.argv[4]) #bunch radius - um (AWAKE: 5.75)
E=float(sys.argv[5]) #bunch emittance - um (AWAKE: 2.0)
N=int(float(sys.argv[6])) #number of macroparticles 

print("#L=%.1f Q=%.1f D=%.2e R=%.1f E=%.1f, N=%d" % (L,Q,D,R,E,N))

ne=Q*1e-12/1.60217646e-19/N #physical number of charges per macroparticle

mpx=300  #mean longitundinal momentum - 1/mc (~150 MeV)
spx=1e-4*mpx #lontudinal momentum spread (0.1%)

#sigma_{x,y,z} (cm)
sx=L*1e-4
sy=sz=R*1e-4

spy=spz=E*1e-4/sy  #transverse momentum spread  - 1/mc


print("# sx = %f" % sx)
print("# sy = %f" % sy)
print("# spx = %f" % spx)
print("# spy = %f" % spy)


""" for gaussian  distributions, we just use the numpy built-in generator """

x =np.random.normal(scale=sx, size=N)
y =np.random.normal(scale=sy, size=N)
z =np.random.normal(scale=sz, size=N)
px=mpx+np.random.normal(scale=spx,size=N)
py=np.random.normal(scale=spy,size=N)
pz=np.random.normal(scale=spz,size=N)

#""" compare actual distribution to desired values """
#print("# %d macroparticles, each corresponding to %.2f physical particles" % (N,ne))
#print("# sx / sx_goal = %f" % (np.std(x)/sx))
#print("# sy / sx_goal = %f" % (np.std(y)/sy))
#print("# sz / sx_goal = %f" % (np.std(z)/sz))
#print("# spx/spx_goal = %f" % (np.std(px)/spx))
#print("# spy/spy_goal = %f" % (np.std(py)/spy))
#print("# spz/spz_goal = %f" % (np.std(pz)/spz))


#np.savetxt("seed.dat",
#           np.c_[x,y,z,px,py,pz,ne*np.ones(N),np.zeros(N)])
#           header="L=%.1f Q=%.1f D=%.2e E=%.1f R=%.1f" % (L,Q,D,E,R))
for n in range(N):
  print(x[n],y[n],z[n],px[n],py[n],pz[n],ne,0)