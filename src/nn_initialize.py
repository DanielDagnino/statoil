#******************************************************************************#
import numpy as np

# # #******************************************************************************#
# from nn_utils import set_mean_and_norm_Nd, set_norm_Nd, set_sqrt_1, set_sqrt_mean_1
# from PIL import Image

#******************************************************************************#
def initialize_parameters_he_cnn( D, FH, FW ):
	
	parameters_nn = {}
	for l in range(len(FW)):
		
		dim_d1 = FW[l]*FH[l]*D[l]
		
		F_aux = np.random.randn(D[l+1],D[l],FH[l],FW[l]).astype("float64") * np.sqrt(2./dim_d1)
		
		parameters_nn['F_'+str(l+1)] = F_aux
		parameters_nn['b_'+str(l+1)] = np.zeros((D[l+1],1)).astype("float64")
	
	return parameters_nn

#******************************************************************************#
def increase_parameters_he_cnn( parameters_nn_old, NN_type, D, FH, FW ):
	
	parameters_nn = {}
	for l in range(len(FW)):
		
		if NN_type[l]=='FC':
			
			dim_d1 = FW[l]*FH[l]*D[l]
			
			T_aux = 0.001*np.random.randn(D[l+1],D[l],FH[l],FW[l]).astype("float64") * np.sqrt(2./dim_d1)
# 			T_aux = np.zeros((D[l+1],D[l],FH[l],FW[l])).astype("float64")
			parameters_nn['F_'+str(l+1)] = T_aux
			parameters_nn['b_'+str(l+1)] = np.zeros((D[l+1],1)).astype("float64")
			
			T_aux = parameters_nn['F_'+str(l+1)]
			I_aux = parameters_nn_old['F_'+str(l+1)]
			T_aux[:I_aux.shape[0],:I_aux.shape[1],:I_aux.shape[2],:I_aux.shape[3]] = I_aux
			
			T_aux = parameters_nn['b_'+str(l+1)]
			I_aux = parameters_nn_old['b_'+str(l+1)]
			T_aux[:I_aux.shape[0],:] = I_aux
			
		else:
			
			parameters_nn['F_'+str(l+1)] = parameters_nn_old['F_'+str(l+1)]
			parameters_nn['b_'+str(l+1)] = parameters_nn_old['b_'+str(l+1)]
		
	return parameters_nn

#******************************************************************************#
def initialize_parameters_reg( Lt, D, regularization, l_reg ):
	
	reg = None
	if regularization:
		reg = np.zeros((Lt))
		for l in range(Lt):
			reg[l] = l_reg/(Lt*D[l+1])
# 			reg[l] = l_reg/(Lt*280)
	
	return reg

#******************************************************************************#
def initialize_adam( D, H, W ):
	
	v = {}
	s = {}
	
	for l in range(len(W)):
		v["dF_"+str(l+1)] = np.zeros((D[l+1],D[l],H[l],W[l])).astype("float64")
		v["db_"+str(l+1)] = np.zeros((D[l+1],1)).astype("float64")
		s["dF_"+str(l+1)] = np.zeros((D[l+1],D[l],H[l],W[l])).astype("float64")
		s["db_"+str(l+1)] = np.zeros((D[l+1],1)).astype("float64")
	
	return v, s

#******************************************************************************#
def increase_initialize_adam( v_old, s_old, NN_type, D, H, W ):
	
	v = {}
	s = {}
	for l in range(len(W)):
		
		if NN_type[l]=='FC':
			
			v["dF_"+str(l+1)] = np.zeros((D[l+1],D[l],H[l],W[l])).astype("float64")
			v["db_"+str(l+1)] = np.zeros((D[l+1],1)).astype("float64")
			s["dF_"+str(l+1)] = np.zeros((D[l+1],D[l],H[l],W[l])).astype("float64")
			s["db_"+str(l+1)] = np.zeros((D[l+1],1)).astype("float64")
			
			T_aux = v['dF_'+str(l+1)]
			I_aux = v_old['dF_'+str(l+1)]
			T_aux[:I_aux.shape[0],:I_aux.shape[1],:I_aux.shape[2],:I_aux.shape[3]] = I_aux
			
			T_aux = v['db_'+str(l+1)]
			I_aux = v_old['db_'+str(l+1)]
			T_aux[:I_aux.shape[0],:] = I_aux
			
			T_aux = s['dF_'+str(l+1)]
			I_aux = s_old['dF_'+str(l+1)]
			T_aux[:I_aux.shape[0],:I_aux.shape[1],:I_aux.shape[2],:I_aux.shape[3]] = I_aux
			
			T_aux = s['db_'+str(l+1)]
			I_aux = s_old['db_'+str(l+1)]
			T_aux[:I_aux.shape[0],:] = I_aux
			
		else:
			
			v['dF_'+str(l+1)] = v_old['dF_'+str(l+1)]
			v['db_'+str(l+1)] = v_old['db_'+str(l+1)]
			s['dF_'+str(l+1)] = s_old['dF_'+str(l+1)]
			s['db_'+str(l+1)] = s_old['db_'+str(l+1)]
		
	return v, s








