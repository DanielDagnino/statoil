#******************************************************************************#
import numpy as np   #@UnusedImport
from scipy.special import expit   #@UnusedImport
import matplotlib.pyplot as plt   #@UnusedImport

#******************************************************************************#
from nn_activation_functions import *   #@UnusedWildImport
from fast_layers import *   #@UnusedWildImport
from fast_layers_nep import *   #@UnusedWildImport
from fast_layers_nep2_DC import *   #@UnusedWildImport
from nn_BN import *   #@UnusedWildImport

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
def L_model_forward_dropout( X, parameters_nn, parameters_design, 
							BN_start=True, BN_update=True, BN_cache=None,
							keep_prob=1.0, pert_neur=0.00, rel_act=0. ):
	
	#------------------------------------------------------------------------------#
	Lc, Drop, DropC, BN, activ, NN_type, _, _, _, _, _, SC, PCW0, PCW1, PCH0, PCH1, \
		FP, SP, PPW0, PPW1, PPH0, PPH1, Pool = parameters_design
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	caches_model = []
	A = X
	
	# Implement CNN.
	for l in range(Lc):
		#--------------------------------------------------------------------------#
		# Convolution.
		
		# Model parameters.
		F = parameters_nn['F_'+str(l+1)]
		b = parameters_nn['b_'+str(l+1)]
# 		print(F.shape)
# 		print(b.shape)
		
		cache_dropC = None
		if DropC[l]:
			cache_dropC = [True]
		
		# Convolution forward.
		conv_param = {'stride': SC[l], 'padH0': PCH0[l], 'padH1': PCH1[l], 'padW0': PCW0[l], 'padW1': PCW1[l]}
# 		X_conv, conv_cache = conv_forward_fast_nep2(A, F_DC, b, conv_param )
		X_conv, conv_cache = conv_forward_fast_nep2_DC(A, F, b, conv_param, DropC[l], pert_neur)
# 		conv_cache = (x, w, b, conv_param, x_cols)
		X = X_conv
		
		#--------------------------------------------------------------------------#
		# Drop-out.
		cache_drop = None
		if Drop[l]:
			drop = np.random.rand(X.shape[1],X.shape[2],X.shape[3])
			drop = drop < keep_prob
			
			for k in range(X.shape[0]):
				X[k,:,:,:] = np.multiply(drop,X[k,:,:,:]) / keep_prob
			
			cache_drop = [drop]
		
		#--------------------------------------------------------------------------#
		# Pooling.
		pool_pad_cache = None
		if Pool[l]:
			X = np.pad(X, ((0, 0), (0, 0), (PPH0[l], PPH1[l]), (PPW0[l], PPW1[l])), mode='constant')
# 	 		x_padded[:, :, padding:-padding, padding:-padding]
			
			pool_param = {'pool_height': FP[l], 'pool_width': FP[l], 'stride': SP[l]}
			X_pool, pool_cache = max_pool_forward_reshape(X, pool_param)
# 	 		cache = (x, x_reshaped, x_pool)
			
			pool_pad_cache = (pool_cache, PPH0[l], PPH1[l], PPW0[l], PPW1[l])
			
			X = X_pool
		
		#--------------------------------------------------------------------------#
		# BN.
		if BN_start:	BN_cache.append(None)
		if BN[l]:
			# Calculate BN.
			if BN_update:	BN_cache[l] = calculate_BN_forward( BN_start, BN_cache[l], NN_type[l], X )
			# Apply BN + Fix pool cache when both Pool and BN is applied.
			X, pool_pad_cache = appply_BN_forward( BN_cache[l], X, pool_pad_cache )
		
		#--------------------------------------------------------------------------#
		# Activation.
		if activ[l]=='R':
			A, activation_cache = relu_forward(X)
		elif activ[l]=='lR':
			A, activation_cache = lrelu_forward(X)
		elif activ[l]=='T':
			A, activation_cache = tanh_forward(X)
		elif activ[l]=='S':
			A, activation_cache = sigmoid_forward(X)
# 	 		activation_cache = g(x)
		else:
			assert False
		
# 		print(str(l)+' min max A = '+str(np.amin(A))+'  '+str(np.amax(A)))
# 		print('   A.mean() = '+str(A.mean()))
# 		print('   np.std(A) = '+str(np.std(A)))
		
		#--------------------------------------------------------------------------#
		# Cache.
		caches_model.append((conv_cache, pool_pad_cache, activation_cache, cache_drop, cache_dropC))
		
	#------------------------------------------------------------------------------#
	return A, caches_model, BN_cache




