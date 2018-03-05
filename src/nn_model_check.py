#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5

#******************************************************************************#
import numpy as np   #@UnusedImport
import os
from sys import exit   #@UnusedImport
import matplotlib.pyplot as plt   #@UnusedImport

#******************************************************************************#
from nn_initialize import *   #@UnusedWildImport
from nn_utils import *   #@UnusedWildImport
from nn_forward_dropout import *   #@UnusedWildImport
from nn_forward_dropout_check import *   #@UnusedWildImport
from nn_backward_dropout import *   #@UnusedWildImport
from nn_grad_test import grads_test   #@UnusedImport
from read_data_set2 import *   #@UnusedWildImport
from nn_BN import *   #@UnusedWildImport

#******************************************************************************#
# Model.
def L_layer_model_check( f_out, augm, check, parameters_design, parameters_read, parameters_min, 
				learning_rate=1e-3, rel_act=0., regularization=True, l_reg=1e-2, 
				num_epochs=1000, n_refresh=4, mini_batch_size=10, 
				initialize=True, parameters_nn=None, 
				seed=10, BN_start=True, BN_update=True, BN_cache=None, keep_prob=0.5, pert_neur=0.02,
				t=0, v=None, s=None, optimizer="adam", beta1=0.9, beta2=0.999, epsilon=1e-8 ):
	
	#------------------------------------------------------------------------------#
	# Some initial data.re
	Lc, _, _, _, _, NN_type, _, _, D, FCW, FCH, _, _, _, _, _, _, _, _, _, _, _, _ = parameters_design
	
	_, _, _, _, _, _, dirpath_data, ind_read_train_set = parameters_read
	
	#------------------------------------------------------------------------------#
	# Parameters initialization.
	if initialize:
		parameters_nn = initialize_parameters_he_cnn( D, FCH, FCW )
	
	if True:
		parameters_nn = increase_parameters_he_cnn( parameters_nn, NN_type, D, FCH, FCW )
		v, s = increase_initialize_adam( v, s, NN_type, D, FCH, FCW )
	
	if regularization:
		reg = initialize_parameters_reg( Lc, D, regularization, l_reg )
	else:
		reg = None
	
	# Initialize the optimizer
	if optimizer == "adam" and v == None and s == None:
		v, s = initialize_adam( D, FCH, FCW )
	
	#------------------------------------------------------------------------------#
	# Optimization loop.
	count1 = 0
	count2 = 0
	iter_refresh = -1
	for _ in range(0, num_epochs):
		
		#--------------------------------------------------------------------------#
		# 
		count1 += 1
		iter_refresh += 1
		
		#--------------------------------------------------------------------------#
		# Read data.
		if iter_refresh%n_refresh==0:
			
			print(' ***** New data set ***** ')
			y_train, e_train, x_train, _, mean_BN, std_BN = read_data_set_BN_augm(augm, ind_read_train_set, dirpath_data, parameters_min, True)
			
			iter_refresh = 0
		
		#--------------------------------------------------------------------------#
		# Define the random minibatches.
		seed += 1
		minibatches = random_mini_batches(x_train, y_train, e_train, mini_batch_size, seed )
		
		#--------------------------------------------------------------------------#
# 		print(' ***** New minibatch ***** '+str(iter_refresh)+' '+str(n_refresh))
		for minibatch in minibatches:
			
			#----------------------------------------------------------------------#
			# Select a minibatch.
			(minibatch_X, minibatch_Y, minibatch_E) = minibatch
			
			#----------------------------------------------------------------------#
			# Forward propagation.
			a_last, caches_forward, BN_cache = L_model_forward_dropout(minibatch_X, parameters_nn, 
															parameters_design, BN_start, BN_update, BN_cache, keep_prob, pert_neur, rel_act)
			BN_start = False
			
			# Compute cost.
			cost, cost_reg = compute_cost(a_last, minibatch_Y, minibatch_E, Lc, parameters_nn, regularization, reg, rel_act)
			
			# Backward propagation.
			grads = L_model_backward_dropout(a_last, minibatch_Y, minibatch_E, parameters_design,
											 caches_forward, BN_cache, keep_prob, pert_neur, rel_act)
			
			#------------------------------------------------------------------------------#
			# Regularization.
			if regularization:
				for l in range(Lc):
					grads['dF'+str(l+1)] += reg[l]*parameters_nn['F_'+str(l+1)]
			
# 			#----------------------------------------------------------------------#
# 			# Gradient test.
# 			ind_grad = []
# 			ind_grad.append( (5, 0, 1, 1) )			# First CNN
# 			for _ in range(5):
# 				ind_grad.append( (30, 30, 1, 1) )	# CNN
# 			ind_grad.append( (100, 100, 5, 5) )		# FC
# 			ind_grad.append( (100, 100, 0, 0) )		# FC
# 			ind_grad.append( (0, 100, 0, 0) )		# Last FC
# 			
# 			grad_test = grads_test( ind_grad, parameters_design, parameters_nn, minibatch_X, 
# 									minibatch_Y, minibatch_E, cost, caches_forward, keep_prob, BN_cache, regularization, reg )
# 			
# 			for l in range(Lc):
# 				id2, id1, ih, iw = ind_grad[l]
# 				print()
# 				g_aux = grad_test[l,0]
# 				dF_aux = grads['dF'+str(l+1)]
# 				print('l = '+str(l))
# 				print('id2, id1, ih, iw = '+str([id2,id1,ih,iw]))
# 				print('Calc: dF = '+str(dF_aux[id2,id1,ih,iw]))
# 				print('Test: dF = '+str(g_aux))
# 				print('F_aux.std  = '+str(np.std(parameters_nn['F_'+str(l+1)])))
# 				print('dF_aux.std = '+str(np.std(dF_aux)))
# 				g_aux = grad_test[l,1]
# 				db_aux = grads['db'+str(l+1)]
# 				print('Calc: db = '+str(db_aux[id2]))
# 				print('Test: db = '+str(g_aux))
# 			
# 			exit(0)
			
			#----------------------------------------------------------------------#
			# Update parameters.
			if optimizer == "gd":
				parameters_nn = update_parameters_with_gd(Lc, parameters_nn, grads, learning_rate)
			elif optimizer == "adam":
				t += 1 # Adam counter
				parameters_nn, v, s = update_parameters_with_adam(Lc, parameters_nn, grads, learning_rate, 
																v, s, t, beta1, beta2, epsilon)
			
			#----------------------------------------------------------------------#
			#----------------------------------------------------------------------#
			cost_new = 0.
			cost_reg_new = 0.
			if check:
				# Forward propagation.
				a_last = L_model_forward_dropout_check(minibatch_X, parameters_nn, parameters_design, caches_forward, BN_cache, keep_prob, pert_neur, rel_act)
				
				# Compute cost.
				cost_new, cost_reg_new = compute_cost(a_last, minibatch_Y, minibatch_E, Lc, parameters_nn, regularization, reg, rel_act)
				
				#----------------------------------------------------------------------#
				# Check performance.
				relat_evol = (cost_new-cost)/cost
				
				if relat_evol>0:   # fail convergence
					coef_corr = -0.5
				else:
					coef_corr = 0.2
				
				if optimizer == "adam":
					parameters_nn = update_parameters_with_adam_check(Lc, parameters_nn, coef_corr*learning_rate, 
																	v, s, t, beta1, beta2, epsilon)
				elif optimizer == "gd":
					parameters_nn = update_parameters_with_gd(Lc, parameters_nn, grads, coef_corr*learning_rate)
				
			#----------------------------------------------------------------------#
			#----------------------------------------------------------------------#
			# 
			count2 += 1
			
			# Print the cost every n_wrt training example
			n_wrt = 1
			if count2 % n_wrt == 0:
				print("e=%i cst %i = %0.3f %0.3f %0.3f %0.3f" % (count1, count2, cost, cost_new, cost_reg, cost_reg_new))
				f_out.write( "%2i %3i %0.3f %0.3f %0.3f %0.3f\n" % (count1, count2, cost, cost_new, cost_reg, cost_reg_new) )
				f_out.flush()
			
# 			if count2 % 100 == 0:
# 				file_path_par_out = "./parameters_nn_tmp.npy"
# 				np.save(os.path.expanduser(file_path_par_out), (parameters_nn, mean_BN, std_BN, BN_cache, t, v, s))
	
	return parameters_nn, t, v, s, mean_BN, std_BN, learning_rate, BN_start, BN_cache

