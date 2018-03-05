#******************************************************************************#
import numpy as np
from scipy.special import expit

#******************************************************************************#
# RELU:
def relu_forward(x):
	relu = x*(x > 0)
	cache = x
	return relu, cache

def relu_backward(coef, cache):
	x = cache
	out = coef*(x > 0)
	return out

#******************************************************************************#
def lrelu_forward(X, a=1e-3):
	out = np.maximum(a * X, X)
	cache = (X, a)
	return out, cache

def lrelu_backward(dout, cache):
	X, a = cache
	dX = dout.copy()
	dX[X < 0] *= a
	return dX

#******************************************************************************#
def sigmoid_forward(x ):
	sigmoid = expit(x )   # expit = 1/(1+exp(-x))
	cache = sigmoid.copy()
	return sigmoid, cache

def sigmoid_backward(coef, cache):
	sigmoid = cache
	out = coef*(sigmoid*(1.-sigmoid))
	return out

#******************************************************************************#
def tanh_forward(x):
	cache = x
	tanh = np.tanh(x)
	return tanh, cache

def tanh_backward(coef, cache):
	x = cache
	out = np.tanh(x)
	dx = coef*(1-out**2)
	return dx

# #******************************************************************************#
# def softmax_loss_forward(x, y):
# 	prob = np.exp(x - np.max(x, axis=1, keepdims=True))
# 	prob /= np.sum(prob, axis=1, keepdims=True)
# 	N = x.shape[0]
# 	cache = (prob.copy(), y)
# 	loss = -np.mean(np.log(prob[np.arange(N), y]))
# 	return loss, cache
# 
# 
# def softmax_loss_backward(dout, cache):
# 	prob, y = cache
# 	N = prob.shape[0]
# 	tmp = np.zeros(prob.shape)
# 	tmp[range(N), y] = 1.0
# 	dx = (prob - tmp)/N
# 	dx = dx * dout
# 	return dx






