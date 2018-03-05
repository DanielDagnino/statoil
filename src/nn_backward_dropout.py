#******************************************************************************#
import numpy as np   #@UnusedImport
from scipy.special import expit   #@UnusedImport

#******************************************************************************#
from config import *   #@UnusedWildImport
from nn_activation_functions import *   #@UnusedWildImport
from fast_layers import *   #@UnusedWildImport
from fast_layers_nep2_DC import *   #@UnusedWildImport
from nn_BN import *   #@UnusedWildImport

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
def L_model_backward_dropout(a_last, Y, E, parameters_design, caches_model, 
							BN_cache, keep_prob=0.5, pert_neur=0.02, rel_act=0.):
	
	#------------------------------------------------------------------------------#
	Lc, _, DropC, _, activ, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = parameters_design
	
	#------------------------------------------------------------------------------#
	grads = {}
# 	print('Y.shape = '+str(Y.shape))
	Y = Y.reshape(a_last.shape)
	E = E.reshape(a_last.shape)
	
	# Initializing the backpropagation
	correct = float((E==True).sum())
	
	a_last = np.maximum(np.minimum(a_last,1.-1e-15),1.e-15)
# 	dA = (-1./correct)*np.divide(coef_cost*Y-a_last+(1.-coef_cost)*np.multiply(Y, a_last), np.multiply(a_last, 1.-a_last) )
	
	mask = np.logical_and( a_last<0.5, (Y==1).reshape(a_last.shape) )
	a_last[mask] = np.maximum(a_last[mask],rel_act)
	mask = np.logical_and( a_last>0.5, (Y==0).reshape(a_last.shape) )
	a_last[mask] = np.minimum(a_last[mask],1.-rel_act)
	dA = (-1./correct)* \
		np.divide( coef_cost*Y-a_last+(1.-coef_cost)*np.multiply(Y, a_last), \
					np.multiply(a_last, 1.-a_last) )
	
	dA[E==False] = 0.
	
	#------------------------------------------------------------------------------#
	# lth layer gradients.
	for l in reversed(range(Lc)):
		
		(conv_cache, pool_pad_cache, activation_cache, cache_drop, _) = caches_model[l]
		if activ[l]=='R':
			dA, grads["dF" + str(l+1)], grads["db" + str(l+1)] = CNN_backward( relu_backward(dA, activation_cache), conv_cache, pool_pad_cache, cache_drop, DropC[l], BN_cache[l], keep_prob, pert_neur)
		elif activ[l]=='lR':
			dA, grads["dF" + str(l+1)], grads["db" + str(l+1)] = CNN_backward( lrelu_backward(dA, activation_cache), conv_cache, pool_pad_cache, cache_drop, DropC[l], BN_cache[l], keep_prob, pert_neur)
		elif activ[l]=='T':
			dA, grads["dF" + str(l+1)], grads["db" + str(l+1)] = CNN_backward( tanh_backward(dA, activation_cache), conv_cache, pool_pad_cache, cache_drop, DropC[l], BN_cache[l], keep_prob, pert_neur)
		elif activ[l]=='S':
			# Lth (last) layer gradients.
			dA, grads["dF" + str(l+1)], grads["db" + str(l+1)] = CNN_backward( sigmoid_backward(dA, activation_cache), conv_cache, pool_pad_cache, cache_drop, DropC[l], BN_cache[l], keep_prob, pert_neur)
		else:
			assert False
		
# 		print('l = '+str(l))
# 		grad = grads["dF" + str(l+1)]
# 		print('grad.shape = '+str(grad.shape))
# 		print('max grad(k)')
# 		gmean = np.mean(np.absolute(grad))
# 		print('gmean = '+str(gmean))
# 		count_rmv = 0
# 		for k in range(grad.shape[0]):
# 			gmax = np.max(np.absolute(grad[k,:,:,:]))
# 			if gmax<0.01*gmean:
# 				print( "%i: %0.5f" % (k,gmax/gmean) )
# 				count_rmv += 1
# 		print('count_rmv/grad.shape[0] = '+str(count_rmv/grad.shape[0]))
		
	return grads

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
def CNN_backward(dA, conv_cache, pool_pad_cache, cache_drop, DropC, BN_cache, keep_prob=0.5, pert_neur=0.02):
	
	#--------------------------------------------------------------------------#
	# BN.
	if BN_cache!=None:
		dA = appply_BN_backward( BN_cache, dA )
	
	#------------------------------------------------------------------------------#
	# 
	if pool_pad_cache!=None:
		(pool_cache, padH0, padH1, padW0, padW1) = pool_pad_cache
		
		dA = max_pool_backward_reshape(dA, pool_cache)
		
		padHmax = max(padH0,padH1)
		padWmax = max(padW0,padW1)
		if padHmax > 0 and padWmax > 0:
			dA = dA[:, :, padH0:-padH1, padW0:-padW1]
		elif padHmax > 0 and padWmax == 0:
			dA = dA[:, :, padH0:-padH1, :]
		elif padWmax > 0 and padHmax == 0:
			dA = dA[:, :, :, padW0:-padW1]
	
	#--------------------------------------------------------------------------#
	# Drop out.
	if cache_drop!=None:
		drop = cache_drop[0]
		for k in range(dA.shape[0]):
			dA[k,:,:,:] = np.multiply(drop,dA[k,:,:,:]) / keep_prob
	
	#------------------------------------------------------------------------------#
	# 
	dA, dF, db = conv_backward_fast_nep2_DC(dA, conv_cache, DropC, pert_neur)
	
	#------------------------------------------------------------------------------#
	return dA, dF, db






