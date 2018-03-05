#******************************************************************************#
import numpy as np

#******************************************************************************#
from config import eps_BN

#******************************************************************************#
def init_BN( NN_type_l, D_l ):
	
	# Max. vaues to save.
	BN_saved_max = 20
	
	# BN zero initialization.
	BN_saved = 0
	
	# Counter.
	counter = np.zeros((BN_saved_max))
	
	# BN over feature map.
	if NN_type_l == 'CN':
		x_sum1 = np.zeros((D_l,BN_saved_max))
		x_sum2 = np.zeros((D_l,BN_saved_max))
	# BN over layer.
	elif NN_type_l == 'FC':
		x_sum1 = np.zeros((BN_saved_max))
		x_sum2 = np.zeros((BN_saved_max))
	else:
		assert False
	
	# BN at this layer.
	BN_cache = ( x_sum1, x_sum2, counter, BN_saved, BN_saved_max, NN_type_l )
	
	return BN_cache


#******************************************************************************#
def calculate_BN_forward( BN_start, BN_cache, NN_type_l, X ):
	
	#------------------------------------------------------------------------------#
	# Initialize BN.
	if BN_start:
		BN_cache = init_BN( NN_type_l, X.shape[1] )
	
	#------------------------------------------------------------------------------#
	# Read BN info.
	x_sum1, x_sum2, counter, BN_saved, BN_saved_max, _ = BN_cache
	
	#------------------------------------------------------------------------------#
	# Calculate new x_sum1 and x_sum2 for all feature maps in the current iteration.
	
	# BN over feature map.
	if NN_type_l == 'CN':
		counter_new = X.shape[0]*X.shape[2]*X.shape[3]
		x_sum1_new = np.zeros((X.shape[1]))
		x_sum2_new = np.zeros((X.shape[1]))
		for d in range(X.shape[1]):
			x_sum1_new[d] = np.sum(X[:,d,:,:])
			x_sum2_new[d] = np.sum(np.power(X[:,d,:,:],2))
	# BN over layer.
	else:
		counter_new = X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]
		x_sum1_new = np.sum(X)
		x_sum2_new = np.sum(np.power(X,2))
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# Max BN saved.
	if BN_saved==BN_saved_max:
		
		#------------------------------------------------------------------------------#
		# Move previous saved BN.
		for bn in range(BN_saved_max-1):
			
			# Counter.
			counter[bn] = counter[bn+1]
			
			# BN over feature map.
			if NN_type_l == 'CN':
				x_sum1[:,bn] = x_sum1[:,bn+1]
				x_sum2[:,bn] = x_sum2[:,bn+1]
			# BN over layer.
			else:
				x_sum1[bn] = x_sum1[bn+1]
				x_sum2[bn] = x_sum2[bn+1]
		
		#------------------------------------------------------------------------------#
		# Last BN saving.
		
		# BN over feature map.
		counter[BN_saved_max-1] = counter_new
		if NN_type_l == 'CN':
			x_sum1[:,BN_saved_max-1] = x_sum1_new[:]
			x_sum2[:,BN_saved_max-1] = x_sum2_new[:]
		# BN over layer.
		else:
			x_sum1[BN_saved_max-1] = x_sum1_new
			x_sum2[BN_saved_max-1] = x_sum2_new
	
	#******************************************************************************#
	# Still saving BN.
	else:
		
		# BN accumulation.
		BN_saved += 1
		
		# Last BN saving.
		
		# Counter.
		counter[BN_saved-1] = counter_new
		
		# BN over feature map.
		if NN_type_l == 'CN':
			x_sum1[:,BN_saved-1] = x_sum1_new[:]
			x_sum2[:,BN_saved-1] = x_sum2_new[:]
		# BN over layer.
		else:
			x_sum1[BN_saved-1] = x_sum1_new
			x_sum2[BN_saved-1] = x_sum2_new
	
	#------------------------------------------------------------------------------#
	# Copy BN info.
	BN_cache = ( x_sum1, x_sum2, counter, BN_saved, BN_saved_max, NN_type_l )
	
	#------------------------------------------------------------------------------#
	return BN_cache


#******************************************************************************#
def appply_BN_forward( BN_cache, X, pool_pad_cache ):
	
	#------------------------------------------------------------------------------#
	# Read BN info.
	x_sum1, x_sum2, counter, _, _, NN_type_l = BN_cache
	
	#------------------------------------------------------------------------------#
	# Pool cache.
	if pool_pad_cache!=None:
		(pool_cache, _, _, _, _) = pool_pad_cache
		(_, x_reshaped, _) = pool_cache
	
	#------------------------------------------------------------------------------#
	# Apply BN backward.
	if NN_type_l == 'CN':
		# BN over feature map.
		for d in range(X.shape[1]):
			mean = np.sum(x_sum1[d,:])/np.sum(counter)
			std = np.sqrt( np.sum(x_sum2[d,:])/np.sum(counter) - np.power(mean,2) + eps_BN )
			X[:,d,:,:] = (X[:,d,:,:]-mean)/std
			# Pool.
			if pool_pad_cache!=None:
				x_reshaped[:,d,:,:,:,:] = (x_reshaped[:,d,:,:,:,:]-mean)/std
# 				x_reshaped[:,d,:,:,:,:] = x_reshaped[:,d,:,:,:,:]/std
	
	# BN over layer.
	else:
		mean = np.sum(x_sum1)/np.sum(counter)
		std  = ((np.sum(counter)-1)/np.sum(counter))*np.sqrt( np.sum(x_sum2)/np.sum(counter) - np.power(mean,2) + eps_BN )
		X = (X-mean)/std
	
	#------------------------------------------------------------------------------#
	return X, pool_pad_cache


#******************************************************************************#
def appply_BN_backward( BN_cache, dA ):
	
	#------------------------------------------------------------------------------#
	# Read BN info.
	x_sum1, x_sum2, counter, _, _, NN_type_l = BN_cache
	
	#------------------------------------------------------------------------------#
	# Apply BN backward.
	if NN_type_l == 'CN':
		# BN over feature map.
		for d in range(dA.shape[1]):
			mean = np.sum(x_sum1[d,:])/np.sum(counter)
			std = np.sqrt( np.sum(x_sum2[d,:])/np.sum(counter) - np.power(mean,2) + eps_BN )
			dA[:,d,:,:] /= std
	
	# BN over layer.
	else:
		mean = np.sum(x_sum1)/np.sum(counter)
		std  = ((np.sum(counter)-1)/np.sum(counter))*np.sqrt( np.sum(x_sum2)/np.sum(counter) - np.power(mean,2) + eps_BN )
		dA /= std
	
	#------------------------------------------------------------------------------#
	return dA




