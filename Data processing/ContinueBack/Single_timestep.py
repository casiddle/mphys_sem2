#!/usr/bin/env python
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
import numpy as np

filename="Data processing\Data\ContinueBack\h5files_driver\v2d_mframe_00001.h5"
try:
  file=h5py.File(filename,'r')
  print("file = %s" % filename)
except IOError:
  sys.stderr.write("File does not exist:\n  %s\n" % filename)
  sys.exit(-1)


file.close()