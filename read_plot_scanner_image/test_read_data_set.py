#******************************************************************************#
import numpy as np
import json
import matplotlib   #@UnusedImport
import matplotlib.pyplot as plt   #@UnusedImport
import scipy.ndimage as ndimage   #@UnusedImport

#******************************************************************************#
from nn_utils import image_rot, set_norm, image_shift, image_perturbation_over, image_roll, image_mirror

#******************************************************************************#
#******************************************************************************#
#******************************************************************************#
# 
def count_data_set(file_labels):
	data = json.load(open(file_labels))
	return data

#******************************************************************************#
# data = count_data_set('/media/daniel/SSD/statoil/train.json')
data = count_data_set('/media/ssd_drive/statoil/train.json')

W = 75
H = 75

# print(str(data[0]["id"]))
# print(str(data[0]["is_iceberg"]))
# print(str(data[0]["inc_angle"]))
# print(str(np.array(data[0]["band_1"]).reshape(H,W).shape))
# print(str(np.array(data[0]["band_2"]).reshape(H,W).shape))

#******************************************************************************#
# Original values in a list.
list_plot = ['2d348f42','e26b8fb0','0859ef5e','2ea3c9f1','b3c50234','1ef4255b',
			'00f3862a','3268312d','256a8dc2','9d128f64','02237e77','30e3bc27','9d1b433b',
			'ebeb858f','5fe2d101','796f23d4','4bed2ea0','c04a768c','8dc762ef','aaef2165',
			'9ff1e0f0','c1bf693f']

nfig = 8
fig = plt.figure(figsize=(80, 80))
ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]

count2 = 0
count3 = 0
# for i in range(nfig*(nfig//4)):
for i in range(len(list_plot)):
	count  = 0
	while data[count]["id"]!=list_plot[i]:	count += 1
	
	im1 = np.array(data[count]["band_1"]).reshape(H,W)
	im2 = np.array(data[count]["band_2"]).reshape(H,W)
	x = np.zeros((H,W,2),dtype="float64")
	x[:,:,0] = im1
	x[:,:,1] = im2
	if True:	x = image_rot( x )
	if False:	x = image_perturbation_over( x )
	if False:	x = image_shift( x )
	if False:	x = image_mirror( x )
	im1 = x[:,:,0] 
	im2 = x[:,:,1]
	
	print(str(data[count]["is_iceberg"]))
	print(str(np.amin(im1))+' '+str(np.amax(im1)))
	print(str(im1.mean())+' '+str(np.std(im1)))
	print(str(np.amin(im2))+' '+str(np.amax(im2)))
	print(str(im2.mean())+' '+str(np.std(im2)))
	
	if data[count]["is_iceberg"]==1:
		ax[count2%8     +int(count2/8)*2*nfig].imshow(im1, interpolation='nearest')
		ax[count2%8+nfig+int(count2/8)*2*nfig].imshow(im2, interpolation='nearest')
		count2 += 1
	else:
		ax[count3%8     +int(count3/8)*2*nfig+4*nfig].imshow(im1, interpolation='nearest')
		ax[count3%8+nfig+int(count3/8)*2*nfig+4*nfig].imshow(im2, interpolation='nearest')
		count3 += 1
	count += 1

for a in ax:
	a.set_xticklabels([])
	a.set_yticklabels([])
	a.set_aspect('auto')

fig.subplots_adjust(wspace=0, hspace=0)

fig.show()
plt.show()


# #******************************************************************************#
# # Original values.
# nfig = 8
# fig = plt.figure(figsize=(80, 80))
# ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]
# 
# count = 0
# for i in range(nfig):
# 	for j in range(nfig//2):
# 		while data[count]["is_iceberg"]==0:	count += 1
# 		
# 		im1 = np.array(data[count]["band_1"]).reshape(H,W)
# 		im2 = np.array(data[count]["band_2"]).reshape(H,W)
# 		
# 		ax[i+j*2*nfig].imshow(im1, interpolation='nearest')
# 		ax[i+nfig+j*2*nfig].imshow(im2, interpolation='nearest')
# 		count += 1
# 
# for a in ax:
# 	a.set_xticklabels([])
# 	a.set_yticklabels([])
# 	a.set_aspect('auto')
# 
# fig.subplots_adjust(wspace=0, hspace=0)
# 
# fig.show()
# plt.show()


# #******************************************************************************#
# # Max correlation.
# nfig = 8
# fig = plt.figure(figsize=(80, 80))
# ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]
# 
# count = 0
# for i in range(nfig):
# 	for j in range(nfig//2):
# 		while data[count]["is_iceberg"]==1:	count += 1
#  		
# 		im1 = np.array(data[count]["band_1"]).reshape(H,W)
# 		im1 = im1/np.amin(np.absolute(im1))
# # 		print('1 max min middle = '+str(np.amax(im1))+' '+str(np.amin(im1))+' '+str(np.amin(im1[37,37])))
# 		im1c = np.zeros_like(im1)
# 		for ii in range(-3,4):
# 			for jj in range(-3,4):
# 				if ii!=0 or jj!=0:
# 					im1b = np.roll(np.roll(im1,ii,axis=0),jj,axis=1)
# 					im1b = np.multiply(im1b,im1)
# 					im1c = np.maximum(im1c,im1b)
# 		im1c = (im1c/np.amax(np.absolute(im1c))).astype("float64")
#  		
# 		im2 = np.array(data[count]["band_2"]).reshape(H,W)
# 		im2 = im2/np.amin(np.absolute(im2))
# # 		print('2 max min middle = '+str(np.amax(im2))+' '+str(np.amin(im2))+' '+str(np.amin(im2[37,37])))
# 		im2c = np.zeros_like(im2)
# 		for ii in range(-3,4):
# 			for jj in range(-3,4):
# 				if ii!=0 or jj!=0:
# 					im2b = np.roll(np.roll(im2,ii,axis=0),jj,axis=1)
# 					im2b = np.multiply(im2b,im2)
# 					im2c = np.maximum(im2c,im2b)
# 		im2c = (im2c/np.amax(np.absolute(im2c))).astype("float64")
#  		
# 		ax[i+j*2*nfig].imshow(im1c, interpolation='nearest', cmap = 'gray')
# 		ax[i+nfig+j*2*nfig].imshow(im2c, interpolation='nearest', cmap = 'gray')
# 		count += 1
#   
# for a in ax:
# 	a.set_xticklabels([])
# 	a.set_yticklabels([])
# 	a.set_aspect('auto')
#  
# fig.subplots_adjust(wspace=0, hspace=0)
#  
# fig.show()
# plt.show()


# #******************************************************************************#
# # Mean correlation.
# nfig = 8
# fig = plt.figure(figsize=(80, 80))
# ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]
#  
# count = 0
# for i in range(nfig):
# 	for j in range(nfig//2):
# 		while data[count]["is_iceberg"]==1:	count += 1
# 		
# 		im1 = np.array(data[count]["band_1"]).reshape(H,W)
# 		im1 = im1/np.amin(np.absolute(im1))
# # 		print('1 max min middle = '+str(np.amax(im1))+' '+str(np.amin(im1))+' '+str(np.amin(im1[37,37])))
# 		im1c = np.zeros_like(im1)
# 		for ii in range(-3,4):
# 			for jj in range(-3,4):
# 				if ii!=0 or jj!=0:
# 					im1b = np.roll(np.roll(im1,ii,axis=0),jj,axis=1)
# 					im1b = np.multiply(im1b,im1)
# 					im1c = im1c+im1b
# 		im1c = (im1c/np.amax(np.absolute(im1c))).astype("float64")
# 		
# 		im2 = np.array(data[count]["band_2"]).reshape(H,W)
# 		im2 = im2/np.amin(np.absolute(im2))
# # 		print('2 max min middle = '+str(np.amax(im2))+' '+str(np.amin(im2))+' '+str(np.amin(im2[37,37])))
# 		im2c = np.zeros_like(im2)
# 		for ii in range(-3,4):
# 			for jj in range(-3,4):
# 				if ii!=0 or jj!=0:
# 					im2b = np.roll(np.roll(im2,ii,axis=0),jj,axis=1)
# 					im2b = np.multiply(im2b,im2)
# 					im2c = im2c+im2b
# 		im2c = (im2c/np.amax(np.absolute(im2c))).astype("float64")
# 		
# 		ax[i+j*2*nfig].imshow(im1c, interpolation='nearest', cmap = 'gray')
# 		ax[i+nfig+j*2*nfig].imshow(im2c, interpolation='nearest', cmap = 'gray')
# 		count += 1
#  
# for a in ax:
# 	a.set_xticklabels([])
# 	a.set_yticklabels([])
# 	a.set_aspect('auto')
# 
# fig.subplots_adjust(wspace=0, hspace=0)
# 
# fig.show()
# plt.show()


# #******************************************************************************#
# # Power 2.
# nfig = 8
# fig = plt.figure(figsize=(80, 80))
# ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]
#  
# count = 0
# for i in range(nfig):
# 	for j in range(nfig//2):
# 		while data[count]["is_iceberg"]==1:	count += 1
# 		im1 = np.array(data[count]["band_1"]).reshape(H,W)
# 		im1 = np.power(im1,2)
# # 		im_sort = np.sort(im1.flatten())
# # 		im_sort = np.amin(im_sort[-35*W:])
# # 		im1[im1>im_sort] = im_sort
#  		
# 		im2 = np.array(data[count]["band_2"]).reshape(H,W)
# 		im2 = np.power(im2,2)
# # 		im_sort = np.sort(im2.flatten())
# # 		im_sort = np.amin(im_sort[-35*W:])
# # 		im2[im2>im_sort] = im_sort
#  		
# 		ax[i+j*2*nfig].imshow(im1, interpolation='nearest', cmap = 'gray')
# 		ax[i+nfig+j*2*nfig].imshow(im2, interpolation='nearest', cmap = 'gray')
# 		count += 1
#  
# for a in ax:
# 	a.set_xticklabels([])
# 	a.set_yticklabels([])
# 	a.set_aspect('auto')
#  
# fig.subplots_adjust(wspace=0, hspace=0)
#  
# fig.show()
# plt.show()


# #******************************************************************************#
# # Mean correlation normalized.
# nfig = 8
# fig = plt.figure(figsize=(80, 80))
# ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]
#   
# count = 0
# eps = 1.e-3
# for i in range(nfig):
# 	for j in range(nfig//2):
# 		while data[count]["is_iceberg"]==0:	count += 1
#   		
# 		im1 = np.array(data[count]["band_1"]).reshape(H,W)
# 		im1 = im1/np.amin(np.absolute(im1))
# 		im1c = np.zeros_like(im1)
# 		im_smooth = ndimage.gaussian_filter(np.absolute(im1), sigma=(5, 5), order=0)
# 		for ii in range(-3,4):
# 			for jj in range(-3,4):
# 				if ii!=0 or jj!=0:
# 					im1b = np.roll(np.roll(im1,ii,axis=0),jj,axis=1)
# 					im1b = np.multiply(im1b,im1)/(im_smooth+eps)
# 					im1c = im1c+im1b
# 		im1c = (im1c/np.amax(np.absolute(im1c))).astype("float64")
#   		
# 		im2 = np.array(data[count]["band_2"]).reshape(H,W)
# 		im2 = im2/np.amin(np.absolute(im2))
# 		im2c = np.zeros_like(im2)
# 		im_smooth = ndimage.gaussian_filter(np.absolute(im2), sigma=(5, 5), order=0)
# 		for ii in range(-3,4):
# 			for jj in range(-3,4):
# 				if ii!=0 or jj!=0:
# 					im2b = np.roll(np.roll(im2,ii,axis=0),jj,axis=1)
# 					im2b = np.multiply(im2b,im2)/(im_smooth+eps)
# 					im2c = im2c+im2b
# 		im2c = (im2c/np.amax(np.absolute(im2c))).astype("float64")
#   		
# 		ax[i+j*2*nfig].imshow(im1c, interpolation='nearest')
# 		ax[i+nfig+j*2*nfig].imshow(im2c, interpolation='nearest')
# 		count += 1
#   
# for a in ax:
# 	a.set_xticklabels([])
# 	a.set_yticklabels([])
# 	a.set_aspect('auto')
#   
# fig.subplots_adjust(wspace=0, hspace=0)
#    
# fig.show()
# plt.show()


# #******************************************************************************#
# # Mean correlation normalized.
# nfig = 8
# fig = plt.figure(figsize=(80, 80))
# ax = [fig.add_subplot(nfig,nfig,i+1) for i in range(nfig*nfig)]
# 
# count = 0
# eps = 1.e-3
# for i in range(nfig):
# 	for j in range(nfig//2):
# 		while data[count]["is_iceberg"]==1:	count += 1
# 		
# 		im1 = np.array(data[count]["band_1"]).reshape(H,W)
# 		im1 = im1/np.amin(np.absolute(im1))
# 		im1 = ndimage.gaussian_filter(im1, sigma=(3, 3), order=0)
# 		
# 		im2 = np.array(data[count]["band_2"]).reshape(H,W)
# 		im2 = im2/np.amin(np.absolute(im2))
# 		im2 = ndimage.gaussian_filter(im2, sigma=(3, 3), order=0)
# 		
# 		ax[i+j*2*nfig].imshow(im1, interpolation='nearest')
# 		ax[i+nfig+j*2*nfig].imshow(im2, interpolation='nearest')
# 		count += 1
# 		
# for a in ax:
# 	a.set_xticklabels([])
# 	a.set_yticklabels([])
# 	a.set_aspect('auto')
# 
# fig.subplots_adjust(wspace=0, hspace=0)
# 
# fig.show()
# plt.show()


