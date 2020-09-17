import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map and 3D surface."""

    print('-------current directory is:', os.getcwd()  )
    print( 'surf_file is saved as:', surf_file)
    print('-----------------------' )
    
    f = h5py.File(surf_file, 'r')

    print('------------STILL ALIVE ------')


print('numpy can be imported')