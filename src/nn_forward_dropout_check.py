#******************************************************************************#
import numpy as np   #@UnusedImport
# from scipy.special import expit   #@UnusedImport
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
def L_model_forward_dropout_check( X, parameters_nn, parameters_design, caches_model,
								 BN_cache, keep_prob=1.0, pert_neur=0.00, rel_act=0. ):
	
	#------------------------------------------------------------------------------#
	Lc, Drop, DropC, BN, activ, _, _, _, _, _, _, SC, PCW0, PCW1, PCH0, PCH1, \
		FP, SP, PPW0, PPW1, PPH0, PPH1, Pool = parameters_design
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	A = X
	
	# Implement CNN.
	for l in range(Lc):
		#--------------------------------------------------------------------------#
		# 
		(_, pool_pad_cache, _, cache_drop, _) = caches_model[l]
		
		#--------------------------------------------------------------------------#
		# Convolution.
		
		# Model parameters.
		F = parameters_nn['F_'+str(l+1)]
		b = parameters_nn['b_'+str(l+1)]
		
		# Convolution forward.
		conv_param = {'stride': SC[l], 'padH0': PCH0[l], 'padH1': PCH1[l], 'padW0': PCW0[l], 'padW1': PCW1[l]}
# 		X_conv, _ = conv_forward_fast_nep2(A, F_DC, b, conv_param)
		X_conv, _ = conv_forward_fast_nep2_DC(A, F, b, conv_param, DropC[l], pert_neur)
		X = X_conv
		
		#--------------------------------------------------------------------------#
		# Drop-out.
		if Drop[l]:
			drop = cache_drop[0]
			for k in range(X.shape[0]):
				X[k,:,:,:] = np.multiply(drop,X[k,:,:,:]) / keep_prob
		
		#--------------------------------------------------------------------------#
		# Pooling.
		if Pool[l]:
			X = np.pad(X, ((0, 0), (0, 0), (PPH0[l], PPH1[l]), (PPW0[l], PPW1[l])), mode='constant')
			
			pool_param = {'pool_height': FP[l], 'pool_width': FP[l], 'stride': SP[l]}
			X_pool, _ = max_pool_forward_reshape(X, pool_param)
			
			X = X_pool
		
		#--------------------------------------------------------------------------#
		# Apply BN + Fix pool cache when both Pool and BN is applied.
		if BN[l]:
			X, _ = appply_BN_forward( BN_cache[l], X, pool_pad_cache )
		
		#--------------------------------------------------------------------------#
		# Activation.
		if activ[l]=='R':
			A, _ = relu_forward(X)
		elif activ[l]=='lR':
			A, _ = lrelu_forward(X)
		elif activ[l]=='T':
			A, _ = tanh_forward(X)
		elif activ[l]=='S':
			A, _ = sigmoid_forward(X)
		else:
			assert False
		
	#------------------------------------------------------------------------------#
	return A




