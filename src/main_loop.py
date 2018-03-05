#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5     /opt/intel/intelpython3/bin/python3.6
# Mac: /opt/intel/intelpython3/bin/python3.6

#******************************************************************************#
import os   #@UnusedImport
import numpy as np   #@UnusedImport
from scipy import ndimage   #@UnusedImport
# import matplotlib   #@UnusedImport
# import matplotlib.pyplot as plt   #@UnusedImport
# import matplotlib.animation   #@UnusedImport

#******************************************************************************#
from read_data_set2 import *   #@UnusedWildImport
from nn_utils import *   #@UnusedWildImport
from define_layers import get_layers
from nn_forward_dropout import *   #@UnusedWildImport
from nn_model_check import L_layer_model_check

#******************************************************************************#
# 
# dirpath_data = '/media/ssd_drive/statoil/train.json'
# dirpath_data = '/media/daniel/SSD/statoil/train.json'
dirpath_data = '../statoil/train.json'
# dirpath_data = '../../statoil/train.json'

#******************************************************************************#
# Random seed initialization.
seed = 5
np.random.seed(seed)

#******************************************************************************#
# # Test laptop fast.
# ne_test			= 2
# ne_train		= 2
# size_mini_train	= 2
# mini_batch_size	= 2
# n_refresh		= 1
# num_epochs		= n_refresh*ne_train//size_mini_train

# # Test laptop fast.
# ne_test			= 12
# ne_train		= 8
# size_mini_train	= 8
# mini_batch_size	= 4
# n_refresh		= 1
# num_epochs		= n_refresh*ne_train//size_mini_train

# # Test laptop fast.
# ne_test			= 12
# ne_train		= 12
# size_mini_train	= 12
# mini_batch_size	= 6
# n_refresh		= 1
# num_epochs		= n_refresh*ne_train//size_mini_train

# # Test laptop hard.
# ne_test			= 0
# ne_train		= 1604-ne_test	# 
# size_mini_train	= ne_train			# 
# mini_batch_size	= 35			# 
# n_refresh		= 1
# num_epochs		= n_refresh*ne_train//size_mini_train

# Test work. 
ne_test			= 0			# 44 24
ne_train		= 1604-ne_test	# 
size_mini_train	= ne_train		# 
mini_batch_size	= 81			# 60 79
n_refresh 		= 1
num_epochs		= n_refresh*ne_train//size_mini_train

print('num_epochs = '+str(num_epochs))

#******************************************************************************#
# Read amount of files and dimension of data.
n_class = 1
ne_all, nx, ny, na = dimensions_data_set( True )

ne_train_0, ne_train_1 =        0, ne_train-1
ne_test_0, ne_test_1   = ne_train, min(ne_train+ne_test-1, ne_all)

ind_read_train_set = range(ne_train_0,ne_train_1+1)
ind_read_test_set  = range(ne_test_0,ne_test_1+1)

print('ne_train_0, ne_train_1 = '+str((ne_train_0, ne_train_1)))
print('ne_all, ne_train, ne_test = '+str(ne_all)+' '+str(ne_train)+' '+str(ne_test))
print('mini_batch_size = '+str(mini_batch_size))

#******************************************************************************#
# CNN design.
Lc, Drop, DropC, BN, activ, NN_type, W, H, D, FCW, FCH, SC, PCW0, PCW1, PCH0, PCH1, \
	FP, SP, PPW0, PPW1, PPH0, PPH1, Pool = get_layers( n_class, nx, ny, na, False )

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
learning_rate = 0.0001
keep_prob = 0.90
pert_neur = 0.02
regularization = True
l_reg = 0.01
n_rep = 50
t = 0   # Adam t.
v, s = None, None
ir0 = 1
check = False
augm_train = 4
augm_test = 4
f_out = open('output.txt','w')
for ir in range(ir0,ir0+n_rep):
	
	#******************************************************************************#
	# Data augmentation + Normalization.
	# parameters_min = ( shift, pertub, mirror, rotate, normalize )
	parameters_min = ( True, True, True, True, True )
# 	parameters_min = ( False, False, False, True )
	
	# 
	p_min = 0.01
# 	rel_act = max(p_min, 0.15-0.05*ir)
	rel_act = max(p_min, 0.20-0.05*ir)
	
	# 
	if ir==0:
		train, initialize = True, True
	else:
		train, initialize = True, False
	
	# Load previous results.
	if initialize:
		parameters_nn, BN_cache = None, []
		BN_start, BN_update = True, True
	else:
		file_path_par_in = "./parameters_nn_"+str(ir)+".npy"
# 		parameters_nn = np.load(os.path.expanduser(file_path_par_in)).item()
		(parameters_nn, mean_BN, std_BN, BN_cache, t, v, s) = np.load(os.path.expanduser(file_path_par_in))
		BN_start, BN_update = False, True
	
	#------------------------------------------------------------------------------#
	# Training.
	
	# Save design of the nn.
	parameters_design = Lc, Drop, DropC, BN, activ, NN_type, W, H, D, FCW, FCH, SC, PCW0, PCW1, PCH0, PCH1, \
	FP, SP, PPW0, PPW1, PPH0, PPH1, Pool
	
	# Save ...
	parameters_read = ne_all, ne_train_0, ne_train_1, nx, ny, na, dirpath_data, ind_read_train_set
	
	#------------------------------------------------------------------------------#
	# Train the model.
	if train:
		# Train.
		seed += 1
		parameters_nn, t, v, s, mean_BN, std_BN, learning_rate, BN_start, BN_cache = L_layer_model_check( 
			f_out, augm_train, check, parameters_design, parameters_read, parameters_min, 
			learning_rate=learning_rate, rel_act=rel_act, regularization=regularization, l_reg=l_reg, num_epochs=num_epochs,
			n_refresh=n_refresh, mini_batch_size=mini_batch_size, initialize=initialize, 
			parameters_nn=parameters_nn, seed=seed, 
			BN_start=BN_start, BN_update=BN_update, BN_cache=BN_cache, 
			keep_prob=keep_prob, pert_neur=pert_neur, t=t, v=v, s=s, optimizer="adam" )
		
		# Save.
		file_path_par_out = "./parameters_nn_"+str(ir+1)+".npy"
		np.save(os.path.expanduser(file_path_par_out), (parameters_nn, mean_BN, std_BN, BN_cache, t, v, s))
	
	#******************************************************************************#
	# Testing.
	if ne_test!=0:
		print()
		
		parameters_min = ( False, False, False, False, True )
		y_test, e_test, x_test, _, _, _ = read_data_set_BN_augm(augm_test, ind_read_test_set, dirpath_data, parameters_min, False, mean_BN, std_BN)
		
		y_prediction_test, _, _ = L_model_forward_dropout(x_test, parameters_nn, parameters_design, False, False, BN_cache, 1., 0.)
		y_prediction_test[e_test==False] = 0.5
		
		cost_test, _ = compute_cost(y_prediction_test, y_test, e_test, Lc, parameters_nn, False, None, 0.)
		
		print("Cost test set = "+str(cost_test))
















