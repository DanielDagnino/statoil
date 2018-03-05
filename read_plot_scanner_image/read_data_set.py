#******************************************************************************#
import numpy as np
import json
import matplotlib   #@UnusedImport
# import matplotlib.pyplot as plt   #@UnusedImport
import scipy.ndimage as ndimage   #@UnusedImport

# #******************************************************************************#
from nn_utils import set_mean_and_norm, image_shift, image_perturbation_over, image_mirror, image_corr   #@UnusedImport
from config import *

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
# 
def dimensions_data_set(train):
	
	if train==True:
		ne = 1604
	else:
		ne = 8424
	
	W  = 75
	H  = 75
	na = deep_selection
	
	return ne, W, H, na

#******************************************************************************#
# 
def read_data_set_BN(ind_read_set, file_name, parameters_min, calc_BN, mean_BN=None, std_BN=None):
	
	# Initial data.
	data = json.load(open(file_name))
	ne_all = len(data)
	
	ne = len(ind_read_set)
	assert ne<=ne_all
	
	W = 75
	H = 75
	D = deep_selection
	
	if calc_BN==False:
		assert np.all(mean_BN)!=None
		assert np.all(std_BN)!=None
	else:
		assert np.all(mean_BN)==None
		assert np.all(std_BN)==None
	
	shift, pertub, mirror, normalize = parameters_min
	
	# Shuffle.
	permutation = list(np.random.permutation(ind_read_set))
# 	permutation = range(len(ind_read_set))
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# Read.
	y = np.zeros((ne,1),dtype="uint16")
	e = np.full((ne,1), True, dtype=bool)
	file_id = ["" for _ in range(ne)]
	x = np.zeros((ne,D,H,W),dtype="float64")
	imag = np.zeros((H,W,D),dtype="float64")
	for kk in range(ne):
		k = permutation[kk]
		y[kk,0] = np.array(data[k]["is_iceberg"])
		file_id[kk] = np.array(data[k]["id"])
		
		#------------------------------------------------------------------------------#
		imag[:,:,0] = np.array(data[k]["band_1"]).reshape(H,W)
		imag[:,:,1] = np.array(data[k]["band_2"]).reshape(H,W)
		if D==4:
			imag[:,:,2] = image_corr( imag[:,:,0] )
			imag[:,:,3] = image_corr( imag[:,:,1] )
		
		if pertub:	imag = image_perturbation_over( imag )
		if shift:	imag = image_shift( imag )
		if mirror:	imag = image_mirror( imag )
# 		if normalize:	imag, _, _ = set_mean_and_norm( imag )
		
		for d in range(D):
			x[kk,d,:,:] = imag[:,:,d]
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# BN.
	if calc_BN:
		imag = np.zeros((H,W,D),dtype="float64")
		mean_BN = np.zeros(D,dtype="float64")
		std_BN  = np.zeros(D,dtype="float64")
		count = 0
		for k in range(ne_all):
			
			#------------------------------------------------------------------------------#
			imag[:,:,0] = np.array(data[k]["band_1"]).reshape(H,W)
			imag[:,:,1] = np.array(data[k]["band_2"]).reshape(H,W)
			if D==4:
				imag[:,:,2] = image_corr( imag[:,:,0] )
				imag[:,:,3] = image_corr( imag[:,:,1] )
			
			#------------------------------------------------------------------------------#
			# 
			for d in range(D):
				mean_BN[d] += np.sum(imag[:,:,d])
				std_BN[d]  += np.sum(np.power(imag[:,:,d],2))
				count += H*W
		
		#------------------------------------------------------------------------------#
		mean_BN /= count
		std_BN = np.sqrt(std_BN/count - np.power(mean_BN,2))
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# BN.
	if normalize:	
		for k in range(ne):
			for d in range(D):
				x[k,d,:,:] = np.divide( x[k,d,:,:]-mean_BN[d], std_BN[d] )
	
	#------------------------------------------------------------------------------#
	return y, e, x, file_id, mean_BN, std_BN


#******************************************************************************#
# 
def read_data_set_BN_test(file_name, parameters_min, calc_BN, mean_BN=None, std_BN=None):
	
	# Initial data.
	data = json.load(open(file_name))
	ne_all = len(data)
	
	W = 75
	H = 75
	D = deep_selection
	
	if calc_BN==False:
		assert np.all(mean_BN)!=None
		assert np.all(std_BN)!=None
	else:
		assert np.all(mean_BN)==None
		assert np.all(std_BN)==None
	
	shift, pertub, mirror, normalize = parameters_min
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# Read.
	file_id = ["" for _ in range(ne_all)]
	x = np.zeros((ne_all,D,H,W),dtype="float64")
	imag = np.zeros((H,W,D),dtype="float64")
	for k in range(ne_all):
		file_id[k] = np.array(data[k]["id"])
		
		#------------------------------------------------------------------------------#
		imag[:,:,0] = np.array(data[k]["band_1"]).reshape(H,W)
		imag[:,:,1] = np.array(data[k]["band_2"]).reshape(H,W)
		if D==4:
			imag[:,:,2] = image_corr( imag[:,:,0] )
			imag[:,:,3] = image_corr( imag[:,:,1] )
		
		if pertub:	imag = image_perturbation_over( imag )
		if shift:	imag = image_shift( imag )
		if mirror:	imag = image_mirror( imag )
# 		if normalize:	imag, _, _ = set_mean_and_norm( imag )
		
		for d in range(D):
			x[k,d,:,:] = imag[:,:,d]
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# BN.
	if calc_BN:
		imag = np.zeros((H,W,D),dtype="float64")
		mean_BN = np.zeros(D,dtype="float64")
		std_BN  = np.zeros(D,dtype="float64")
		count = 0
		for k in range(ne_all):
			file_id[k] = np.array(data[k]["id"])
			
			#------------------------------------------------------------------------------#
			imag[:,:,0] = np.array(data[k]["band_1"]).reshape(H,W)
			imag[:,:,1] = np.array(data[k]["band_2"]).reshape(H,W)
			if D==4:
				imag[:,:,2] = image_corr( imag[:,:,0] )
				imag[:,:,3] = image_corr( imag[:,:,1] )
			
			for d in range(D):
				x[k,d,:,:] = imag[:,:,d]
			
			#------------------------------------------------------------------------------#
			# 
			for d in range(D):
				mean_BN[d] += np.sum(imag[:,:,d])
				std_BN[d]  += np.sum(np.power(imag[:,:,d],2))
				count += H*W
		
		#------------------------------------------------------------------------------#
		mean_BN /= count
		std_BN = np.sqrt(std_BN/count - np.power(mean_BN,2))
	
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------#
	# BN.
	if normalize:	
		for k in range(ne_all):
			for d in range(D):
				x[k,d,:,:] = np.divide( x[k,d,:,:]-mean_BN[d], std_BN[d] )
	
	#------------------------------------------------------------------------------#
	return x, file_id









