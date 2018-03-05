#******************************************************************************#
import numpy as np   #@UnusedImport

#******************************************************************************#
from nn_utils import *   #@UnusedWildImport
from nn_forward_dropout_check import *   #@UnusedWildImport

#******************************************************************************#
# Model.
def grads_test( ind_grad, parameters_design, parameters_nn, 
				minibatch_X, minibatch_Y, minibatch_E, 
				cost, caches_model, keep_prob, BN_cache, regularization, reg ):
	
	#------------------------------------------------------------------------------#
	# Some initial data.
	Lc, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = parameters_design
	
	# delta.
	delta = 1.e-5
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	grad_test = np.zeros((Lc,2))
	for l in range(Lc):
		
# 		print(l)
		id2, id1, ih, iw = ind_grad[l]
		
		for k in range(2):
			
			#------------------------------------------------------------------------------#
			if k==0:
				F_aux = parameters_nn['F_'+str(l+1)]
				deltaPar = delta*np.amax(np.absolute(F_aux))
# 				deltaPar = delta
				F_aux[id2,id1,ih,iw] = F_aux[id2,id1,ih,iw] + deltaPar
				parameters_nn['F_'+str(l+1)] = F_aux
			elif k==1:
				b_aux = parameters_nn['b_'+str(l+1)]
				deltaPar = delta
				b_aux[id2,0] = b_aux[id2,0] + deltaPar
				parameters_nn['b_'+str(l+1)] = b_aux
			
			#------------------------------------------------------------------------------#
			# 
			# Forward propagation.
			a_last = L_model_forward_dropout_check(minibatch_X, parameters_nn, parameters_design, caches_model, BN_cache, keep_prob)
# 			print(str(l)+' min max a_last = '+str(np.amin(a_last))+'  '+str(np.amax(a_last)))
			
			# Compute cost.
			cost_delta, _ = compute_cost(a_last, minibatch_Y, minibatch_E, Lc, parameters_nn, regularization, reg)
			
			grad_test[l,k] = (cost_delta-cost)/deltaPar
			
			#------------------------------------------------------------------------------#
			if k==0:
				F_aux[id2,id1,ih,iw] = F_aux[id2,id1,ih,iw] - deltaPar
				parameters_nn['F_'+str(l+1)] = F_aux
			elif k==1:
				b_aux[id2,0] = b_aux[id2,0] - deltaPar
				parameters_nn['b_'+str(l+1)] = b_aux
			
	return grad_test




