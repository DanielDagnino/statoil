#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5

#******************************************************************************#
import os   #@UnusedImport
import numpy as np   #@UnusedImport
from scipy import ndimage   #@UnusedImport
import matplotlib   #@UnusedImport
import matplotlib.pyplot as plt   #@UnusedImport
import matplotlib.animation   #@UnusedImport

#******************************************************************************#
from read_data_set import *   #@UnusedWildImport
from nn_utils import *   #@UnusedWildImport
from define_layers import get_layers
from nn_model_check import L_layer_model_check
from nn_forward_dropout import *   #@UnusedWildImport

#******************************************************************************#
# Labels of the final layer.
n_class = 1

# 
# dirpath_data = '/media/ssd_drive/statoil/test.json'
dirpath_data = '/media/daniel/SSD/statoil/train.json'

# Read amount of files and dimension of data.
ne_all, nx, ny, na = dimensions_data_set(True)
print('nx, ny, na = '+str(nx)+' '+str(ny)+' '+str(na))

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
# CNN design.
Lt, Lc, Drop, NN_type, W, H, D, FCW, FCH, SC, PCW0, PCW1, PCH0, PCH1, \
    FP, SP, PPW0, PPW1, PPH0, PPH1, Pool = get_layers( n_class, nx, ny, na, False )

#******************************************************************************#
# Load previous results.
file_path_par_in = "./parameters_nn_tmp.npy"
parameters_nn = np.load(os.path.expanduser(file_path_par_in)).item()

#******************************************************************************#
# 
l, nf, d = 0, 0, 0
Fc = parameters_nn['F_'+str(l+1)]

z = np.array([Fc[nf,d,i,j] for j in range(Fc.shape[2]) for i in range(Fc.shape[3])])

nx = Fc.shape[2]
ny = Fc.shape[3]
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)

X, Y = np.meshgrid(x, y)
Z = z.reshape(ny, nx)

plt.pcolor(X, Y, Z)
plt.show()








