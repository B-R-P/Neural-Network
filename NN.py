"""
Neural Network
--------------
Neural network made from scratch for learning purpose.

Note:
	Some equation used are made up thus it may or may not work.

LICENSE
-------

MIT License

Copyright (c) 2020 Bijin Regi Panicker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from zipfile import ZipFile
from math import e,sqrt
from numpy import dot,sum,array
from tqdm import tqdm
from threading import Thread
from numba import jit
from random import randrange

@jit(nopython=True,cache=True,fastmath=True)
def wbgen(count,inum):
	"""For random generation of weights at intialization"""
	b = sigmoid(count)
	if .5<b:
		b = 1-b
	w = array([1/((sqrt(inum)+.21)*randrange(1,8)) for i in range(inum)])
	return w,b	

class Neuron():
	"""Fundamental part of a neural network"""
	count = 0
	def __init__(self,inum,new,activation):
		if new:# Random generation of weights and bias
			Neuron.count+=1
			self.w,self.b=wbgen(self.count, inum)
			self.berror=0
		self.activation  = activation.lower()
	def input(self,inp):
		"""Input array of values to neuron and computing its output"""
		self.inp = inp
		# Output of neuron => o = f(W . I + B)  
		#		  | Activation fn      ||     dot product   | bias   |
		self.o = actdic[self.activation](dot(self.w,self.inp)+self.b)
	def updateweight(self,learning_rate,momentum):
		"""Learning happens here => changinge weights and bias"""
		self.berror,self.w,self.b = uw(self.error,self.berror,learning_rate,momentum,self.inp,self.w,self.b)

@jit(nopython=True,fastmath=True,nogil=True,cache=True)
def uw(error,berror,learning_rate,momentum,inp,w,b):
	"""Adjust weights and bias"""
	error*=learning_rate/sqrt(len(w))
	error+=berror*momentum
	for wi,ip in enumerate(inp):
		w[wi]+=error*ip
	return error,w,b+error

#Activation functions(JIT enabled)
reLU = jit(nopython=True,cache=True,nogil=True)(lambda n:n if 0<n else 0)
sigmoid  = jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda n:1/(1+(e**-n)))
softmax= jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda x:e**x)
tanh= jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda x:1-(2/(1+e**(2*x))))
actdic = { # Dictionary of activation functions
	"sigmoid":lambda o:sigmoid(o),
	"relu": lambda o:reLU(o),
	"softmax": lambda o:softmax(o),
	"tanh": lambda o:tanh(o)
}
transfer_derivative = jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda output:output*(1-float(output)))

def save_model(Layers,name):
	"""Save trained model as a zipfile with textual representations"""
	zipfile = ZipFile(f"{name}.model","w")
	# Representing network structure
	zipfile.writestr("ModelInfo",f"{len(Layers[0])},{len(Layers[1:-1])},{len(Layers[1])},{len(Layers[-1])},{Layers[1][0].activation}")
	for i,layer in enumerate(Layers,1):
		txt=""
		for neuron in layer:
			# Representing weights and bias of a layer
			stxt = str(list(neuron.w))+str(neuron.b)+","+str(neuron.berror)+"\r\n"
			for j in "float(') [":
				stxt=stxt.replace(j,"")
			txt+=stxt
		zipfile.writestr(f"Layer{i}",txt)
	zipfile.close()

def regain_model(name):
	"""Read model saved as a text into network"""
	zipfile = ZipFile(f"{name}.model",'r')
	minfo=zipfile.read("ModelInfo").decode("utf-8").split(",")
	Layers = Network(*([int(i) for i in minfo[:-1]]+[False,minfo[-1]]))
	for i,layer in enumerate(Layers,1):
		file = zipfile.read(f"Layer{i}").decode("utf-8").splitlines()
		for l,neuron in zip(file,layer):
			# Reimplementing the layer of network as represented in text form
			l = l.split("]")
			neuron.w  = array([float(w) for w in l[0].split(",")])
			sp = l[1].split(",")
			neuron.b = float(sp[0])
			neuron.berror=float(sp[1])
	zipfile.close()
	return Layers

@jit(nopython=False,cache=True,fastmath=True,nogil=True)
def smax(l):
	"Softmax function"
	m = max(l)
	l = [i-m for i in l]
	t = sum(l)
	return [i/t for i in l]
#Giving input
def ForwardPropagation(inp,Layers):
	"""Ouputs are computed as per """
	threads=[]# Multi-threaded execution
	# Inputing to neuron in first layer
	for neuron,i in zip(Layers[0],inp):
		p=Thread(target=neuron.input,args=(array([i]),))
		threads.append(p)
		p.start()
	for thread in threads:
		thread.join()
	#Forward Propagation
	for llayer,layer in  zip(Layers,Layers[1:]):
		ni = [float(n.o) for n in llayer]#Output from previous layer 
		if llayer[0].activation=='softmax':
			ni = smax(ni)
		threads=[]
		ni=array(ni)
		# Passing outputs into next layer of neurons
		for neuron in layer:
			p=Thread(target=neuron.input,args=(ni,))
			threads.append(p)
			p.start()
		for thread in threads:
			thread.join()
	return Layers

def fun(bl,i,neuron,learning_rate,momentum):
	"""Propagate error from one layer of another"""
	neuron.error = 0
	for bneuron in bl:
		neuron.error+=bneuron.w[i]*bneuron.error
	neuron.error*=transfer_derivative(neuron.o)
	neuron.updateweight(learning_rate,momentum)

def funf(neuron,expected,learning_rate,momentum):
	"""Propagate error from first layer"""
	output = neuron.o
	err = (expected-output)
	neuron.error=err*abs(err)*transfer_derivative(output)
	neuron.updateweight(learning_rate,momentum)

def bpropagation(out,Layers,learning_rate,momentum=0):
	"""Propagate error through a neuron and adjust its weights and bias"""
	#Backward Propagation
	Layers.reverse()
	threads=[]
	# Training first layer
	for neuron,expected in zip(Layers[0],out):
		p = Thread(target=funf,args=(neuron,expected,learning_rate,momentum,))
		threads.append(p)
		p.start()
	for thread  in threads:
		thread.join()
	# Training all layers
	for bl,layer in zip(Layers,Layers[1:]):
		threads=[]
		for i,neuron in enumerate(layer):
			p = Thread(target=fun,args=(bl,i,neuron,learning_rate,momentum,))
			threads.append(p)
			p.start()
		for thread  in threads:
			thread.join()
	Layers.reverse()
	return Layers
	"""Backprogration happens here"""

def predict(inp,Layers):
	"""Compute output from trained network"""
	ForwardPropagation(inp,Layers)
	opl = [i.o for i in Layers[-1]]
	m = max(opl)
	return (opl.index(m),m,opl)

def printer(Layers):
	"""Print outputs of each layer"""
	#Printing value of neurons
	for layer in Layers:
		print([str(n.o) for n in layer])
		print()

def wprinter(Layers):
	"""Print weight and bias of neurons"""
	for layer in Layers:
		print([[list(map(str,i.w)),str(i.b)] for i in layer])
		print()

# To train with single array of input and output
train  = lambda inp,out,learning_rate,Layers,momentum=0:bpropagation(out,ForwardPropagation(inp,Layers),float(learning_rate),momentum)

def traina(Layers,inp,out,learning_rate,momentum=0,iteration=1,pdisplay=False,clr=True):
	""" Train neurons by reducing error between inputs(inp) and expected outputs(out)"""
	lr = float(learning_rate)
	momentum=float(momentum)
	k=lr/iteration
	c = list(zip([array(i) for i in inp],[array(i) for i in out]))
	bar=tqdm(range(iteration))
	for j in bar:
		if pdisplay:
			if clr:
				print("Iteration:",j+1,"Learning Rate:",lr)
			else:
				print("Iteration:",j+1)
		for ip,op in c:
			Layers=train(ip,op,lr,Layers,momentum=momentum)
		if clr:
			lr-=k# To show details in progress bar
			bar.set_postfix_str(str(round(lr,3)))
	return Layers

#sn-No of starting neuron
#hl-No of hidden layers
#hn-No of neuron in hidden layers
#on-No of output neuron
def Network(sn,hl,hn,on,new=True,activation="reLU"):
	"""
	Generate multiple layers of neurons
	"""
	Layers = []
	#Making Layers
	#Input Layers
	Layers.append([])
	#Hidden Layers
	for i in range(hl):
		Layers.append([])
	#Output Layers
	Layers.append([])
	#Adding neurons to layer
	#Input Layers
	for i in range(sn):
		Layers[0]+= [Neuron(1,new,activation="tanh")]
	#Hidden Layers
	for i in range(hn):
		Layers[1]+=[Neuron(sn,new,activation=activation)]
	for layer in Layers[2:-1]:
		for i in range(hn):
			layer+=[Neuron(hn,new,activation=activation)]
	#Output Layers
	for i in range(on):
		Layers[-1]+=[Neuron(hn,new,activation="sigmoid")]
	return Layers

def model(sn,hl,hn,on,model_name,activation="reLU"):
	"""
	Recover trained model if exist otherwise create new one
	"""
	try:
		Layers = regain_model(model_name)
	except FileNotFoundError:
		print("Making",model_name)
		Layers  = Network(sn,hl,hn,on,activation=activation)
		save_model(Layers, model_name)
	return Layers

def check_err(Layers,ipa,opa):
	"""Compute of error between computed output and expected output"""
	err=0
	ol=len(opa)
	for inp,out in zip(ipa,opa):
		for i in range(ol):
			err+=abs(ForwardPropagation(inp,Layers)[-1][i].o-out[i])
	return err