#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5
# Mac-Centos: /opt/intel/intelpython3/bin/python3.6

#******************************************************************************#
import os   #@UnusedImport
import numpy as np   #@UnusedImport
from scipy import ndimage   #@UnusedImport
import matplotlib   #@UnusedImport
import matplotlib.pyplot as plt   #@UnusedImport

#******************************************************************************#
n_test = 2
n_test = 6
n_test = 11
# n_test = 3?
# n_test = 5?
# n_test = 8?

n_test = 2
cost_reg = 0.01
# file_name = "output.txt"
# file_name = "/home/ddagnino/cluster/PROCESSED_DATA/dagnino/test"+str(n_test)+"/output (copy).txt"
file_name = "/home/ddagnino/cluster/PROCESSED_DATA/dagnino/test"+str(n_test)+"/output.txt"
data = np.loadtxt(file_name)

#------------------------------------------------------------------------------#
s = data[:,2]

window_len = 1
w = np.ones(window_len,'d')
y1 = np.convolve(w/w.sum(),s,mode='valid')-cost_reg

window_len = 20
w = np.ones(window_len,'d')
y2 = np.convolve(w/w.sum(),s,mode='valid')-cost_reg

window_len = 40
w = np.ones(window_len,'d')
y3 = np.convolve(w/w.sum(),s,mode='valid')-cost_reg

#------------------------------------------------------------------------------#
# Plot the cost.
fig, ax = plt.subplots()

x = np.linspace(1,y1.shape[0],num=y1.shape[0])
ax.plot(x,y1)

x = np.linspace(1,y2.shape[0],num=y2.shape[0])
plt.plot(x,y2)

x = np.linspace(1,y3.shape[0],num=y3.shape[0])
plt.plot(x,y3)

plt.xlim(1,y1.shape[0])
plt.ylim(0.15,0.60)

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 81))

ax.set_yticks(np.arange(0.05, 0.60, 0.05))
ax.set_yticks(np.arange(0.05, 0.60, 0.01), minor=True)

# ax.grid(True)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Cost evolution of test "+str(n_test))

plt.show()

#******************************************************************************#








