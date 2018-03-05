#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5
# Mac: /opt/intel/intelpython3/bin/python3.6

#******************************************************************************#
import os   #@UnusedImport
import numpy as np   #@UnusedImport

#******************************************************************************#
from read_data_set2 import *   #@UnusedWildImport
from define_layers import get_layers
from nn_forward_dropout import *   #@UnusedWildImport
from nn_utils import *   #@UnusedWildImport

#******************************************************************************#
# 
# dirpath_data = '/media/daniel/SSD/statoil/train.json'
dirpath_data = '/media/ssd_drive/statoil/train.json'
# dirpath_data = '../statoil/train.json'

#******************************************************************************#
# Labels of the final layer.
n_class = 1

# Read amount of files and dimension of data.
ne_all, nx, ny, na = dimensions_data_set( True )

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
# CNN design.
Lc, Drop, DropC, BN, activ, NN_type, W, H, D, FCW, FCH, SC, PCW0, PCW1, PCH0, PCH1, \
	FP, SP, PPW0, PPW1, PPH0, PPH1, Pool = get_layers( n_class, nx, ny, na, False )

#******************************************************************************#
# Save design of the nn.
parameters_design = Lc, Drop, DropC, BN, activ, NN_type, W, H, D, FCW, FCH, SC, PCW0, PCW1, PCH0, PCH1, \
	FP, SP, PPW0, PPW1, PPH0, PPH1, Pool

#******************************************************************************#
# Results.
# (parameters_nn, mean_BN, std_BN, BN_cache) = np.load(os.path.expanduser('./parameters_nn_tmp.npy')).item()
(parameters_nn, mean_BN, std_BN, BN_cache, t, v, s) = \
	np.load(os.path.expanduser('./parameters_nn_T1.npy'))
BN_start, BN_update = False, False

f1 = open("./results_all.txt", 'w')
f1.write('id,is_iceberg\n')

mini_batch_size = 24
parameters_min = ( False, False, False, True )

# ind_read_test_set = range(1604)

ne_test         = 24
ne_train        = 1604-ne_test
ne_train_0, ne_train_1 =        0, ne_train-1
ne_test_0, ne_test_1   = ne_train, min(ne_train+ne_test-1, ne_all)
ind_read_test_set  = range(ne_test_0,ne_test_1+1)
ind_read_train_set = range(ne_train_0,ne_train_1+1)

# y_test, e_test, x_test, file_id, _, _ = read_data_set_BN(ind_read_train_set, dirpath_data, 
# 															parameters_min, False, mean_BN, std_BN)

y_test, e_test, x_test, file_id, _, _ = read_data_set_BN(ind_read_test_set, dirpath_data, 
															parameters_min, False, mean_BN, std_BN)

minibatches = mini_batches_test2(x_test, y_test, e_test, file_id, mini_batch_size )

#--------------------------------------------------------------------------#
# 
count = 0
cost = 0
for minibatch in minibatches:
	(minibatch_X, minibatch_Y, minibatch_E, minibatch_file_id) = minibatch
	
	y_prediction, _, _ = L_model_forward_dropout(minibatch_X, parameters_nn, parameters_design, 
												False, False, BN_cache, 1.)
	
# 	y_prediction[y_prediction>0.999] = 0.999
# 	y_prediction[y_prediction<0.001] = 0.001
	
	cost += compute_cost_no_norm(y_prediction, minibatch_Y, minibatch_E)
	
	# Print to file.
	for ie in range(y_prediction.shape[0]):
		f1.write(str(minibatch_file_id[ie]))
		f1.write(","+"{:.6f}".format(float(np.squeeze(y_prediction[ie,0]))))
		f1.write(","+"{:.6f}".format(float(np.squeeze(y_test[ie,0]))))
		if np.absolute(y_prediction[ie,0]-y_test[ie,0])>0.5:
			f1.write("   WRONG!!!")
		elif np.absolute(y_prediction[ie,0]-y_test[ie,0])>0.3:
			f1.write("   psss 3!!!")
		elif np.absolute(y_prediction[ie,0]-y_test[ie,0])>0.2:
			f1.write("   psss 2")
		f1.write("\n")
		f1.flush()
		count += 1
	
# 	print('Cost acumulated (count='+str(count)+') = '+str(cost/count))
	print('Cost acmltd '+str(count)+' = '+str(cost/count))

f1.close()

print('End!!!')










