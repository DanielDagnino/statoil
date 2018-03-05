#!/usr/bin/python

#******************************************************************************#
def new_size( W, F, S, P0, P1 ):
	if not (W+P0+P1-F)%S == 0:
		print( ' *** Error: W,F,S,P0,P1 = '+str(W)+' '+str(F)+' '+str(S)+' '+str(P0)+' '+str(P1) )
		print( ' *** Error: W+P0+P1-F = '+str(W+P0+P1-F) )
		print( ' *** Error: (W+P0+P1-F)%S = '+str((W+P0+P1-F)%S) )
		raise AssertionError()
	Wo = (W+P0+P1-F)//S + 1
	return Wo

#******************************************************************************#
# 
def get_layers( n_class, nx, ny, na, print_lay ):
	
	# 
	W = []
	H = []
	
	# 
	W.append(nx)
	H.append(ny)
	
	# 
	T = True
	F = False
	N = None
	
	#------------------------------------------------------------------------------#
	# Layers.
	Lc = 3*2+3*1+1
	
	# CNN or FC (as CNN).
	NN_type = [   'CN','CN','CN','CN','CN','CN', 'FC', 'FC',  'FC',  'FC']
# 	activ = [           'lR',   'lR',  'lR',  'lR',  'lR',  'lR',    'T',   'T',    'T',    'S' ]
	activ = [           'lR',   'lR',  'lR',  'lR',  'lR',  'lR',   'lR',  'lR',  'lR',    'S' ]
# 	activ = [           'R',   'R',  'R',  'R',  'R',  'R',   'R',  'R',    'S' ]
	assert activ[Lc-1]=='S'	# Last layer = Sigmoid.
	Drop  = [        F,   F,   F,   F,   F,   F,    F,    T,    T,    T ]
# 	DropC = [        F,   F,   F,   F,   F,   F,    F,    F,    F,    F ]
# 	DropC = [        F,   T,   F,   T,   F,   T,    T,    T,    T ]
	DropC = [        T,   T,   T,   T,   T,   T,    T,    T,    T,    T ]
	BN   = [         T,   T,   T,   T,   T,   T,    T,    T,    T,    F ]
	assert BN[Lc-1]==F	# Last layer (final probability).
	FCW = [          3,   3,   3,   3,   3,   3,   10,    1,    1,    1 ]
	FCH = [          3,   3,   3,   3,   3,   3,   10,    1,    1,    1 ]
# 	D   = [    na,  64,  64, 128, 128, 256, 256, 4096, 4096,    n_class ]
# 	D   = [    na,  64,  64, 128, 128, 256, 256, 1024, 1024, 1024,    n_class ]
# 	D   = [    na,  64,  64, 128, 128, 256, 256, 1280, 1280, 1280,    n_class ]
# 	D   = [    na,  64,  64, 128, 128, 256, 256, 2048, 2048, 2048,    n_class ]
	D   = [    na,  64,  64, 128, 128, 256, 256, 1536, 1536, 1536,    n_class ]
	SC  = [          1,   1,   1,   1,   1,   1,    1,    1,    1,    1 ]
	PCW0 = [         1,   1,   1,   1,   1,   1,    0,    0,    0,    0 ]
	PCW1 = PCW0
	PCH0 = PCW0
	PCH1 = PCW0
	# Pooling.
	Pool = [         F,   T,   F,   T,   F,   T,     F,    F,    F,    F ]
	FP   = [         N,   2,   N,   2,   N,   2,     N,    N,    N,    N ]
	SP   = FP
	PPH0 = [         N,   0,   N,   0, 	 N,   0,     N,    N,    N,    N ]
	PPH1 = [         N,   1,   N,   0,   N,   1,     N,    N,    N,    N ]
	PPW0 = PPH0
	PPW1 = PPH1
	
	#------------------------------------------------------------------------------#
	# 
	param_tot = 0
	if print_lay: 
		print(' W = '+str(W))
		print(' H = '+str(H))
	
	#------------------------------------------------------------------------------#
	for l in range(Lc):
		
		#--------------------------------------------------------------#
		if print_lay: print('Conv')
		
		Wnew = new_size( W[l],  FCW[l],  SC[l],  PCW0[l],  PCW1[l] )
		if Pool[l]==False: W.append( Wnew )
		if print_lay:
			print(' W = '+str(W))
			print(' Wnew = '+str(Wnew))
		
		Hnew = new_size( H[l],  FCH[l],  SC[l],  PCH0[l],  PCH1[l] )
		if Pool[l]==False: H.append( Hnew )
		if print_lay:
			print(' H = '+str(H))
			print(' Hnew = '+str(Hnew))
		
		param = D[l+1]*D[l]*(1+FCH[l]*FCW[l])
		param_tot = param_tot + param
		
		if print_lay: print(' #Parameters = '+str(param))
		
		#--------------------------------------------------------------#
		if Pool[l]==True:
			
			if print_lay: print('Pool')
			
			Wnew = new_size( W[l], FP[l], SP[l], PPW0[l], PPW1[l] )
			W.append( Wnew )
			if print_lay: print(' W = '+str(W))
			
			Hnew = new_size( H[l], FP[l], SP[l], PPH0[l], PPH1[l] )
			H.append( Hnew )
			if print_lay: print(' H = '+str(H))
		
	#------------------------------------------------------------------------------#
	print()
	print(' ***** Final ***** ')
	print('Lc = '+str(Lc))
	print('W  = '+str(W))
	print('H  = '+str(H))
	print('D  = '+str(D))
	print('#Total Parameters (with bias terms) / 1e6 = '+str(param_tot/1e6))
	
	#------------------------------------------------------------------------------#
	# 
	return Lc, Drop, DropC, BN, activ, NN_type, W, H, D, FCW, FCH, SC, PCW0, PCW1, PCH0, PCH1, FP, SP, \
		PPW0, PPW1, PPH0, PPH1, Pool

# #******************************************************************************#
# # Check 1.
# get_layers( 1, 75, 75, 2, True )








