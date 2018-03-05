#******************************************************************************#
import numpy as np
import math
# from scipy import misc
import scipy

#******************************************************************************#
from config import *   #@UnusedWildImport

#******************************************************************************#
def compute_cost(AL, Y, E, Lt, parameters_nn, regularization, reg, rel_act):
	
	# Cost function.shift
# 	print('compute_cost Y.shape  = '+str(Y.shape))
# 	print('compute_cost E.shape  = '+str(E.shape))
# 	print('compute_cost AL.shape = '+str(AL.shape))
	assert Y.shape[0] == AL.shape[0]
	assert Y.shape[1] == AL.shape[1]
	
	AL = np.maximum(np.minimum(AL,1.-1e-15),1.e-15)
	
	mask = np.logical_and( AL<0.5, (Y==1).reshape(AL.shape) )
	AL[mask] = np.maximum(AL[mask],rel_act)
	mask = np.logical_and( AL>0.5, (Y==0).reshape(AL.shape) )
	AL[mask] = np.minimum(AL[mask],1.-rel_act)
	
	aux = coef_cost*np.multiply(Y.flatten(), np.log(AL).flatten()) + \
		np.multiply(1.-Y.flatten(), np.log(1.-AL).flatten())
	
	cost_reg = 0.
	if regularization:
		for l in range(Lt):
			cost_reg += 0.5*reg[l]*np.sum(np.power(parameters_nn['F_'+str(l+1)],2))
	
	aux = aux.reshape(Y.shape)
	aux[E==False] = 0.
	
	correct = float((E==True).sum())
	cost = (-1./correct)*np.sum( aux ) + cost_reg
	
	cost = np.squeeze(cost)
	assert cost.shape == ()
	
	return cost, cost_reg

#******************************************************************************#
def compute_cost_no_norm(AL, Y, E, rel_act):
	
	# Cost function.shift
	assert Y.shape[0] == AL.shape[0]
	assert Y.shape[1] == AL.shape[1]
	
	AL = np.maximum(np.minimum(AL,1.-1e-15),1.e-15)
	
# 	mask = np.logical_and( AL<0.5, (Y==1).reshape(AL.shape) )
# 	AL[mask] = np.maximum(AL[mask],rel_act)
# 	mask = np.logical_and( AL>0.5, (Y==0).reshape(AL.shape) )
# 	AL[mask] = np.minimum(AL[mask],1.-rel_act)
	
	aux = coef_cost*np.multiply(Y.flatten(), np.log(AL).flatten()) + \
		np.multiply(1.-Y.flatten(), np.log(1.-AL).flatten())
	
	aux = aux.reshape(Y.shape)
	aux[E==False] = 0.
	
	cost = -np.sum( aux )
	
	cost = np.squeeze(cost)
	assert cost.shape == ()
	
	return cost

#******************************************************************************#
def compute_cost_L2(AL, Y, E):
	
	# Cost function.
	assert Y.shape[0] == AL.shape[0]
	assert Y.shape[1] == AL.shape[1]
	
	aux = np.square( Y.flatten() - AL.flatten() )
	aux = aux.reshape(Y.shape)
	aux[E==False] = 0.
	cost = np.mean(aux)
	
	cost = np.squeeze(cost)
	assert cost.shape == ()
	
	return cost

# #******************************************************************************#
# def predict(X, parameters_nn, parameters_design):
# 	
# 	from nn_forward import L_model_forward
# 	
# 	A, _ = L_model_forward(X, parameters_nn, parameters_design)
# 	
# 	y_predictions = np.round(A)
# 	
# 	return y_predictions

#******************************************************************************#
def random_mini_batches(X, Y, E, mini_batch_size = 2, seed = 0):
	
	np.random.seed(seed)
	m = X.shape[0]			# number of training examples
	mini_batches = []
	
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]
	shuffled_E = E[permutation,:]
	
	# Step 2: Partition (shuffled_X, shuffled_Y, shuffled_E). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size,:]
		mini_batch_E = shuffled_E[k * mini_batch_size:(k + 1) * mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_E)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
#		end = m - mini_batch_size * math.floor(m / mini_batch_size)
		_ = m - mini_batch_size * math.floor(m / mini_batch_size)
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:,:]
		mini_batch_E = shuffled_E[num_complete_minibatches * mini_batch_size:,:]
		mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_E)
		mini_batches.append(mini_batch)
	
	return mini_batches

#******************************************************************************#
def mini_batches_test2(X, Y, E, file_id, mini_batch_size = 2):
	
	m = X.shape[0]			# number of training examples
	mini_batches = []
	
	# Step 2: Partition (shuffled_X, shuffled_Y, shuffled_E). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = X[k * mini_batch_size:(k + 1) * mini_batch_size,:,:,:]
		mini_batch_Y = Y[k * mini_batch_size:(k + 1) * mini_batch_size,:]
		mini_batch_E = E[k * mini_batch_size:(k + 1) * mini_batch_size,:]
		mini_batch_file_id = file_id[k * mini_batch_size:(k + 1) * mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_E, mini_batch_file_id)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
#		end = m - mini_batch_size * math.floor(m / mini_batch_size)
		_ = m - mini_batch_size * math.floor(m / mini_batch_size)
		mini_batch_X = X[num_complete_minibatches * mini_batch_size:,:,:,:]
		mini_batch_Y = Y[num_complete_minibatches * mini_batch_size:,:]
		mini_batch_E = E[num_complete_minibatches * mini_batch_size:,:]
		mini_batch_file_id = file_id[num_complete_minibatches * mini_batch_size:]
		mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_E, mini_batch_file_id)
		mini_batches.append(mini_batch)
	
	return mini_batches

#******************************************************************************#
def mini_batches_test(X, file_id, mini_batch_size = 2):
	
	m = X.shape[0]			# number of training examples
	mini_batches = []
	
	# Step 2: Partition (X, shuffled_Y, shuffled_E). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = X[k * mini_batch_size:(k + 1) * mini_batch_size,:,:,:]
		mini_batch_file_id = file_id[k * mini_batch_size:(k + 1) * mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_file_id)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
#		end = m - mini_batch_size * math.floor(m / mini_batch_size)
		_ = m - mini_batch_size * math.floor(m / mini_batch_size)
		mini_batch_X = X[num_complete_minibatches * mini_batch_size:,:,:,:]
		mini_batch_file_id = file_id[num_complete_minibatches * mini_batch_size:]
		mini_batch = (mini_batch_X, mini_batch_file_id)
		mini_batches.append(mini_batch)
	
	return mini_batches


#******************************************************************************#
def image_shift( data ):
	
	plf = 2*(np.random.random_sample() < 0.5)-1
	px = np.random.random_sample()
	py = np.random.random_sample()
	
	na = data.shape[2]
	for i in range(na):
		#
		data_aux = data[:,:,i]
		
		#
		ix_shift = 10
		ix_shift = int(plf*px*ix_shift)
		
		iy_shift = 10
		iy_shift = int(py*iy_shift)
		
		# 
		data_aux = np.roll(data_aux,ix_shift,axis=0)
		data_aux = np.roll(data_aux,iy_shift,axis=1)
		
		# 
		data[:,:,i] = data_aux
		
	#----------------------------------------------------------------------#
	return data

#******************************************************************************#
def image_roll( data ):
	
	px = np.random.random_sample()
	py = np.random.random_sample()
	
	na = data.shape[2]
	ix_lim = data.shape[0]
	iy_lim = data.shape[1]
	for i in range(na):
		#
		data_aux = data[:,:,i]
		
		#
		ix_shift = ix_lim
		ix_shift = int(px*ix_shift)
		
		#
		iy_shift = iy_lim
		iy_shift = int(py*iy_shift)
		
		#
		data_aux = np.roll(data_aux,ix_shift,axis=0)
		data_aux = np.roll(data_aux,iy_shift,axis=1)
		
		# 
		data[:,:,i] = data_aux
	
	return data

#******************************************************************************#
# X (ne, nf, h, w)
def set_mean_and_norm_per_filter( X ):
#	print('X.shape = '+str(X.shape))
	nf = X.shape[1]
	std = np.zeros(nf)
	mean = np.zeros(nf)
	for i in range(nf):
		aux = X[:,i,:,:]
		
		mean[i] = aux.mean()
#		mean[i] = 0.
		aux = aux-mean
		
		std[i] = np.std(aux)
#		std[i] = 1.
		aux = aux/std[i]
		
		X[:,i,:,:] = aux
		
	return X, mean, std

#******************************************************************************#
def set_norm( data ):
	
	nl = data.shape[2]
	std = np.zeros(nl)
	for i in range(nl):
		aux = data[:,:,i]
		std[i] = np.std(aux)
		if std[i] != 0:
			data[:,:,i] = data[:,:,i]/std[i]
	
	return data, std

#******************************************************************************#
# X (ne, nf, h, w)
def set_mean_and_norm_per_filter_test( X, mean, std ):
	
	nf = X.shape[1]
	for i in range(nf):
		aux = X[:,i,:,:]
		
		aux = aux-mean
		
		aux = aux/std[i]
		
		X[:,i,:,:] = aux
		
	return X

#******************************************************************************#
# dF (nf, d, h, w) or (d[l+1], d[l], h, w)
def set_mean_and_norm_per_filter_back_kernel_F( dF, std ):
	
#	print('dA.shape = '+str(dA.shape))
#	print('std.shape = '+str(std.shape))
	nf = dF.shape[0]
	for i in range(nf):
		dF[i,:,:,:] = std[i]*dF[i,:,:,:]
	
	return dF

#******************************************************************************#
def set_mean_and_norm_per_filter_back_kernel_b( db, std ):
	
#	print('db.shape = '+str(db.shape))
#	print('std.shape = '+str(std.shape))
	nf = db.shape[0]
	for i in range(nf):
		db[i] = std[i]*db[i]
	
	return db

#******************************************************************************#
def set_mean_and_norm( data ):
	
	nl = data.shape[2]
	std = np.zeros(nl)
	mean = np.zeros(nl)
	for i in range(nl):
		mean[i] = data[:,:,i].mean()
		data[:,:,i] = data[:,:,i]-mean[i]
		
		std[i] = np.std(data[:,:,i])
		data[:,:,i] = data[:,:,i]/std[i]
	
	return data, mean, std

#******************************************************************************#
def set_mean_and_norm_Nd( data ):
	
	mean = data.mean()
	data = data-mean
	
	std = np.std(data)
	data = data/std
	
	return data

#******************************************************************************#
def set_norm_Nd( data ):
	
	std = np.std(data)
	data = data/std
	
	return data

#******************************************************************************#
def set_sqrt_1( data ):
	
	norm = np.sum(np.power(data,2))
	data = data/norm
	
	return data

#******************************************************************************#
def set_sqrt_mean_1( data ):
	
	mean = data.mean()
	data = data-mean
	
	norm = np.sum(np.power(data,2))
	data = data/norm
	
	return data

#******************************************************************************#
def update_parameters_with_gd( Lc, parameters_nn, grads, learning_rate ):
	
	# 
	( parameters_cnn ) = parameters_nn
	
	# CNN layers.
	for l in reversed(range(Lc)):
# 		if l==0 or l==1 or l==2:
# 			grads['dF'+str(l+1)] = np.zeros_like(grads['dF'+str(l+1)])
		parameters_nn['F_'+str(l+1)] = parameters_nn['F_'+str(l+1)] - learning_rate*grads['dF'+str(l+1)]
		db = parameters_cnn['b_'+str(l+1)]
		db = db - learning_rate*grads['db'+str(l+1)].reshape(db.shape)
	
	return parameters_nn

#******************************************************************************#
def update_parameters_with_adam(Lc, parameters_nn, grads, learning_rate, v, s, t,
								beta1=0.9, beta2=0.999, epsilon=1e-8):
	
	# Update rule for each parameter.
	# Perform Adam update on all parameters
	v_corrected = {}						 # Initializing first moment estimate, python dictionary
	s_corrected = {}						 # Initializing second moment estimate, python dictionary
	
	# CNN layers.
	for l in reversed(range(Lc)):
		
		# Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
		v['dF_'+str(l+1)] = beta1*v['dF_'+str(l+1)] + (1.-beta1)*grads['dF'+str(l+1)]
		v['db_'+str(l+1)] = beta1*v['db_'+str(l+1)] + (1.-beta1)*grads['db'+str(l+1)].reshape(v['db_'+str(l+1)].shape)
		
		# Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
		v_corrected['dF_'+str(l+1)] = v['dF_'+str(l+1)]/(1.-np.power(beta1, t))
		v_corrected['db_'+str(l+1)] = v['db_'+str(l+1)]/(1.-np.power(beta1, t))
		
		# Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
		s['dF_'+str(l+1)] = beta2*s['dF_'+str(l+1)] + (1.-beta2)*np.power(grads['dF'+str(l+1)], 2)
		s['db_'+str(l+1)] = beta2*s['db_'+str(l+1)] + (1.-beta2)*np.power(grads['db'+str(l+1)].reshape(s['db_'+str(l+1)].shape), 2)
		
		# Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
		s_corrected['dF_'+str(l+1)] = s['dF_'+str(l+1)]/(1.-np.power(beta2, t))
		s_corrected['db_'+str(l+1)] = s['db_'+str(l+1)]/(1.-np.power(beta2, t))
		
		# Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
		parameters_nn['F_'+str(l+1)] = parameters_nn['F_'+str(l+1)] - learning_rate*v_corrected['dF_'+str(l+1)] / (np.sqrt(s_corrected['dF_'+str(l+1)]) + epsilon)
		parameters_nn['b_'+str(l+1)] = parameters_nn['b_'+str(l+1)] - learning_rate*v_corrected['db_'+str(l+1)] / (np.sqrt(s_corrected['db_'+str(l+1)]) + epsilon)
		
# 		print('l = '+str(l))
# 		FFF = parameters_nn['F_'+str(l+1)]
# 		print('FFF.shape = '+str(FFF.shape))
# 		print('max F')
# 		FFFmean = np.mean(np.absolute(FFF))
# 		print('FFFmean = '+str(FFFmean))
# 		count_rmv = 0
# 		for k in range(FFF.shape[0]):
# 			FFFmax = np.max(np.absolute(FFF[k,:,:,:]))
# 			if FFFmax<0.01*FFFmean:
# 				print( "%i: %0.5f" % (k,FFFmax/FFFmean) )
# 				count_rmv += 1
# 		print('count_rmv/FFF.shape[0] = '+str(count_rmv/FFF.shape[0]))
		
	return parameters_nn, v, s

#******************************************************************************#
def update_parameters_with_adam_check(Lc, parameters_nn, learning_rate, v, s, t,
								beta1=0.9, beta2=0.999, epsilon=1e-8):
	
	# Update rule for each parameter.
	# Perform Adam update on all parameters
	v_corrected = {}						 # Initializing first moment estimate, python dictionary
	s_corrected = {}						 # Initializing second moment estimate, python dictionary
	
	# CNN layers.
	for l in reversed(range(Lc)):
		
		# Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
		v_corrected['dF_'+str(l+1)] = v['dF_'+str(l+1)]/(1.-np.power(beta1, t))
		v_corrected['db_'+str(l+1)] = v['db_'+str(l+1)]/(1.-np.power(beta1, t))
		
		# Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
		s_corrected['dF_'+str(l+1)] = s['dF_'+str(l+1)]/(1.-np.power(beta2, t))
		s_corrected['db_'+str(l+1)] = s['db_'+str(l+1)]/(1.-np.power(beta2, t))
		
		# Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
		parameters_nn['F_'+str(l+1)] = parameters_nn['F_'+str(l+1)] - learning_rate*v_corrected['dF_'+str(l+1)] / (np.sqrt(s_corrected['dF_'+str(l+1)]) + epsilon)
		parameters_nn['b_'+str(l+1)] = parameters_nn['b_'+str(l+1)] - learning_rate*v_corrected['db_'+str(l+1)] / (np.sqrt(s_corrected['db_'+str(l+1)]) + epsilon)
	
	return parameters_nn

#******************************************************************************#
def image_perturbation_over( data ):
	
	#----------------------------------------------------
	px = 1.-0.1*np.random.random_sample()
	py = 1.-0.1*np.random.random_sample()
	
	for i in range(data.shape[2]):
		
		data_aux = data[:,:,i]
		
		data_aux2 = scipy.ndimage.interpolation.zoom( data_aux, (px,py) )
		
		idiff = data_aux.shape[0]-data_aux2.shape[0]
		ia = idiff//2
		ib = data_aux.shape[0]-(idiff-ia)
		
		idiff2 = data_aux.shape[1]-data_aux2.shape[1]
		ic = idiff2//2
		idd = data_aux.shape[1]-(idiff2-ic)
		
		data_aux[ia:ib,ic:idd] = data_aux2
		
		data[:,:,i] = data_aux
	
	#----------------------------------------------------------------------#
	return data


#******************************************************************************#
def image_perturbation_extend( data ):
	
	from scipy import misc
	
	#----------------------------------------------------
	px = 1.-0.1*np.random.random_sample()
	py = 1.-0.1*np.random.random_sample()
	
	for i in range(data.shape[2]):
		
		data_aux = data[:,:,i]
		
		data_aux2 = misc.imresize( data_aux, (int(px*data_aux.shape[0]),int(py*data_aux.shape[1])) )
		
		idiff = data_aux.shape[0]-data_aux2.shape[0]
		ia = idiff//2
		ib = data_aux.shape[0]-(idiff-ia)
		
		idiff2 = data_aux.shape[1]-data_aux2.shape[1]
		ic = idiff2//2
		idd = data_aux.shape[1]-(idiff2-ic)
		
		data_aux = np.zeros_like(data_aux)
		data_aux[ia:ib,ic:idd] = data_aux2
		
		data_aux = np.zeros_like(data_aux)
		data_aux[ia:ib,ic:idd] = data_aux2
		
		data_aux[0:ia,:] = data_aux[ia,:]
		aux = data_aux[:,ic]
		data_aux[:,0:ic] = aux[:,np.newaxis]
		
		if ib>0:
			data_aux[-(idiff-ia):,:] = data_aux[-(idiff-ia)-1,:]
		
		if idd>0:
			aux = data_aux[:,-(idiff2-ic)-1]
			data_aux[:,-(idiff2-ic):] = aux[:,np.newaxis]
		
		data[:,:,i] = data_aux
	
	#----------------------------------------------------------------------#
	return data


#******************************************************************************#
def image_mirror( data ):
		
	p1 = np.random.random_sample()
	p2 = np.random.random_sample()
	for i in range(data.shape[2]):
		
		data_aux = data[:,:,i]
		
		if p1>0.5:
			data_aux = np.fliplr(data_aux)
		
		if p2>0.5:
			data_aux = np.flipud(data_aux)
		
		data[:,:,i] = data_aux
	
	#----------------------------------------------------------------------#
	return data

#******************************************************************************#
def image_corr( im1 ):
	
	import scipy.ndimage as ndimage   #@UnusedImport
	eps = 1.e-3
	im1 = im1/np.amin(np.absolute(im1))
	im1c = np.zeros_like(im1)
	im_smooth = ndimage.gaussian_filter(np.absolute(im1), sigma=(5, 5), order=0)
	for ii in range(-3,4):
		for jj in range(-3,4):
			if ii!=0 or jj!=0:
				im1b = np.roll(np.roll(im1,ii,axis=0),jj,axis=1)
				im1b = np.multiply(im1b,im1)/(im_smooth+eps)
				im1c = im1c+im1b
	im1c = (im1c/np.amax(np.absolute(im1c))).astype("float64")
	
	#----------------------------------------------------------------------#
	return im1c

#******************************************************************************#
def image_rot( data ):
	
	from scipy import ndimage
	
	angle = 360.*np.random.random_sample()-180.
	for i in range(data.shape[2]):
		
		data_aux = data[:,:,i]
		
		cval = 10000.
		data_rot = ndimage.rotate(data_aux, angle, reshape=False, cval=cval)
		mask = data_rot==cval
		data_rot[mask] = data_aux[mask]
		
		data[:,:,i] = data_rot
	
	#----------------------------------------------------------------------#
	return data






















