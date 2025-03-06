#!/usr/bin/env python

# VLP3 - Virtual Laser Plasma Post-Processing
# Developed by John Farmer
# This version was edited by Carys Siddle
# Calculates various beam parameters from a particles.h5 file
# usage:  em.py <directory> <save number> <species>

#
# For time evolution, combine with a bash script to loop over files

import numpy
import h5py
import sys
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dir', type=str, nargs=1)
parser.add_argument('saves', type=int, nargs='+')
parser.add_argument('species', type=int, nargs=1)
parser.add_argument('-e','--emtype', type=str, nargs=1, choices=['v','g','p'], default="v",help="Method to use for the emittance calculation.  Options are:\n  v (default)- calculate the velocity assuming gamma=px; g - calculate the velocity using the full gamma calculation; p - calculate the emittance using the momentum (no need to normalize at the end)")
parser.add_argument('-o','--outputformat', type=str, nargs=1, choices=['t','h','l','H'], default="t",help="Format for output.  Options are:\n t (default)- tabular; l - list; h - as list, but with comment header identiying columns; H - header only")
parser.add_argument('-u','--unit', type=str, nargs=1, choices=['u','c','k'], default="u",help="Unit for output.  Options are:\n u (default)- micrometer; c - centimetre; k - plasma units (k_p)")
parser.add_argument('-w','--weight', action='store_true',help="Print total weight only and exit")
parser.add_argument('-t','--time', action='store_true',help="Print phase and exit")

args = parser.parse_args()

stride=1 # sets the stride for sampling particles


if args.outputformat[0] in ["H","h"]:
  if args.emtype[0]=="p":
    sys.stdout.write('#%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ("phase", "wmy", "wmz", "wmpy", "wmpz", "stdev_y", "stdev_z", "stdev_py", "stdev_pz", "em_y", "em_z", "wmg", "stdev_g", "N", "W") )
  else:
    sys.stdout.write('#%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ("phase", "wmy", "wmz", "wmvy", "wmvz", "stdev_y", "stdev_z", "stdev_vy", "stdev_vz", "em_y", "em_z", "wmg", "stdev_g", "N", "W") )
if args.outputformat[0] == "H":
  exit(0)

#read variables from command line
dir=args.dir[0]
dumpn=args.saves[0]
species=args.species[0]

#Particle dataset
try:
  filename="%s/h5files/vs%s_3d_particles.h5" % (dir,str(dumpn).zfill(3))
  p1=h5py.File(filename,'r')
except Exception:
  sys.stderr.write("Could not open particles %s\n" % filename)
  exit(-1)

#if "phase" in p1.keys():
phase=p1["phase"][0]
#else:
#  phase=dumpn

if args.time:
  print("Phase = ",phase)
  exit(0)

wavelength=p1['PlasmaWavelength'][0]

#load particle data

mean=numpy.zeros((7))
meansq=numpy.zeros((5))
W=0
N=0

for key in p1:
  if key[:15] == ("Ions_of_sort_%s" % str(species).zfill(2) ):
    i1=p1[key]
    
    w=numpy.array(i1['weight'])
    N_l=w.size
    N=N+N_l

    W_l=numpy.sum(w)
    Wo=W
        
    W=W+W_l
    
    if args.weight or N_l==0:
      continue
    
    y=i1['y']
    z=i1['z']
    py=i1['py']
    pz=i1['pz']
    px=i1['px']

    if args.emtype[0]=="g":
      gamma=(1+px**2+py**2+pz**2)**0.5
    else:
      gamma=px
    if args.emtype[0]!="p":
      py/=gamma
      pz/=gamma
    
    d=numpy.array([y,z,py,pz,gamma,y*py,z*pz])

    
    #mean_l=numpy.sum(w*d,axis=1)/W_l
    #meansq_l=numpy.sum(w*d[:5,]**2,axis=1)/W_l
    
    mean=mean*(Wo/W)+numpy.sum(w*d,axis=1)/W
    meansq=meansq*(Wo/W)+numpy.sum(w*d[:5,]**2,axis=1)/W


if N==0:
  sys.stderr.write("No particles of species %d in file %s\n" % (species, filename) )
  exit(-1)
  
charge=W*1.609e-19
if (charge>1e-6):
  charge_string="%g C" % charge
elif (charge>1e-9):
  charge_string="%g nC" % (charge*1e9)
elif (charge>1e-12):
  charge_string="%g pC" % (charge*1e12)
elif (charge>1e-15):
  charge_string="%g fC" % (charge*1e15)
else:
  charge_string="%g C" % charge

if args.weight:
  print("Phase  = %d\n" % int(phase))
  print("Particles      %16s%16s%16s" % ("physical","macro","charge"))
  print("               %16g%16g%16s" % (W,N,charge_string))
  exit(0)
  
#calculate weighted mean of y,z, vy,vz
if args.unit[0]=="u":
  unitconv=1e4
  unitname="um"
elif args.unit[0]=="c":
  unitconv=1
  unitname="cm"
elif args.unit[0]=="k":
  unitconv=2*3.14159/wavelength
  unitname="kp"

wmy = mean[0]*unitconv
wmz = mean[1]*unitconv
wmpy = mean[2]
wmpz = mean[3]

wmg = mean[4]


stdev_y=(meansq[0]-mean[0]**2)**0.5*unitconv
stdev_z=(meansq[1]-mean[1]**2)**0.5*unitconv
stdev_py=(meansq[2]-mean[2]**2)**0.5
stdev_pz=(meansq[3]-mean[3]**2)**0.5

var_g=(meansq[4]-mean[4]**2)
if var_g<0:
  sys.stderr.write("Variance in gamma < 0 (%f): setting stddev to 0\n" % var_g)
  stdev_g=0
else:
  stdev_g=var_g**0.5

em_y=((meansq[0]-mean[0]**2)*(meansq[2]-mean[2]**2)-(mean[5]-mean[0]*mean[2])**2)**0.5*unitconv
em_z=((meansq[1]-mean[1]**2)*(meansq[3]-mean[3]**2)-(mean[6]-mean[1]*mean[3])**2)**0.5*unitconv

if args.emtype[0] != "p":
  em_y*=wmg
  em_z*=wmg

if(args.outputformat[0] != "t"):
    sys.stdout.write('%g %g %g %g %g %g %g %g %g %g %g %g %g %d %g\n' % (phase, wmy, wmz, wmpy, wmpz, stdev_y, stdev_z, stdev_py, stdev_pz, em_y, em_z, wmg, stdev_g, N, W) )
else:
    # Calculation of energy
    if species == 1:
      energy = wmg * 938.08e6  # Convert gamma to energy (MeV)
    else:
      energy = wmg * 0.511e6 # Convert gamma to energy (MeV)
    if energy < 1e3:
        energy_string = "%g eV" % energy
    elif energy < 1e6:
        energy_string = "%g keV" % (energy / 1e3)
    elif energy < 1e9:
        energy_string = "%g MeV" % (energy / 1e6)
    elif energy < 1e12:
        energy_string = "%g GeV" % (energy / 1e9)
    else:
        energy_string = "%g TeV" % (energy / 1e12)

    # New output for energy, gamma, and spread
    energy=energy/1e9
    print(f"Energy: {energy} ")  # Prints the energy in a human-readable format
    print(f"Gamma: {wmg}")  # Prints the gamma value formatted to 2 decimal places
    spread_percentage = 100 * stdev_g / wmg if wmg != 0 else 0  # Calculate the spread in percentage
    print(f"Spread: {spread_percentage:.2f}")  # Prints the spread formatted to 2 decimal places with a percentage sign
    print(f"Geometric Emittance: {(em_y*em_z)**0.5}")
    print(f"Y Emittance:{em_y}")
    print(f"Z Emittance:{em_z}")
    print(f"Beam Radius:{0.5*(stdev_y+stdev_z)}")


  

  
  #-----

  # print("Phase  = %d\n" % int(phase))
  # print("\nBeam           %16s%16s%16s" % ("energy","gamma","spread"))
  # print("               %16s%16g%15g%%" % (energy_string,wmg,100*stdev_g/wmg))
  
  # print("\nParticles      %16s%16s%16s" % ("physical","macro","charge"))
  # print("               %16g%16g%16s" % (W,N,charge_string))

  # print("\nPosition (%s)  %16s%16s" % (unitname,"y","z"))
  # print("  mean         %16g%16g" % (wmy,wmz))
  # print("  std dev      %16g%16g" % (stdev_y,stdev_z))
  # if args.emtype[0]=="p":
  #   print("\nMomentum (1/mc)%16s%16s" % ("y","z"))
  # else:
  #   print("\nVelocity (1/c) %16s%16s" % ("y","z"))
  # print("  mean         %16g%16g" % (wmpy,wmpz))
  # print("  std dev      %16g%16g" % (stdev_py,stdev_pz))
  # print("\nEmittance (%s) %16s%16s%16s" % (unitname,"y","z","geometric"))
  # print("               %16g%16g%16g" % (em_y,em_z,(em_y*em_z)**0.5))
  

