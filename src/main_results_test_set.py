#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5

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
dirpath_data = '/media/ssd_drive/statoil/test.json'
# dirpath_data = '/media/daniel/SSD/statoil/test.json'
# dirpath_data = '../statoil/test.json'

#******************************************************************************#
# Labels of the final layer.
n_class = 1

# Read amount of files and dimension of data.
ne_all, nx, ny, na = dimensions_data_set( False )

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
(parameters_nn, mean_BN, std_BN, BN_cache, t, v, s) = np.load(os.path.expanduser('./parameters_nn_T1.npy'))
BN_start, BN_update = False, False

f1 = open("./results.txt", 'w')
f1.write('id,is_iceberg\n')

mini_batch_size = 50
parameters_min = ( False, False, False, False, True )

x_test, file_id = read_data_set_BN_test(dirpath_data, parameters_min, False, mean_BN, std_BN)

minibatches = mini_batches_test(x_test, file_id, mini_batch_size )

#--------------------------------------------------------------------------#
# 
count = 0
for minibatch in minibatches:
	print('count = '+str(count))
	(minibatch_X, minibatch_file_id) = minibatch
	
	y_prediction, _, _ = L_model_forward_dropout(minibatch_X, parameters_nn, parameters_design, False, False, BN_cache, 1.)
	
	# Print to file.
	for ie in range(y_prediction.shape[0]):
		f1.write(str(minibatch_file_id[ie]))
		f1.write(","+"{:.6f}".format(float(np.squeeze(y_prediction[ie,0])))+"\n")
		count += 1
	f1.flush()

f1.close()
print('End!!!')




