from zipfile import ZipFile
from math import e,sqrt
from numpy import dot,sum,array
from tqdm import tqdm
from threading import Thread
from numba import jit
from random import randrange
@jit(nopython=True,cache=True,fastmath=True)
def wbgen(count,inum):
	b = sigmoid(count)
	if .5<b:
		b = 1-b
	w = array([1/((sqrt(inum)+.21)*randrange(1,8)) for i in range(inum)])
	return w,b	
class Neuron():
	count = 0
	def __init__(self,inum,new,activation):
		if new:
			Neuron.count+=1
			self.w,self.b=wbgen(self.count, inum)
			self.berror=0
		self.activation  = activation.lower()
	def input(self,inp):
		self.inp = inp
		self.o = actdic[self.activation](dot(self.w,self.inp)+self.b)
	def updateweight(self,learning_rate,momentum):
		 self.berror,self.w,self.b = uw(self.error,self.berror,learning_rate,momentum,self.inp,self.w,self.b)
@jit(nopython=True,fastmath=True,nogil=True,cache=True)
def uw(error,berror,learning_rate,momentum,inp,w,b):
	error*=learning_rate/sqrt(len(w))
	error+=berror*momentum
	for wi,ip in enumerate(inp):
		w[wi]+=error*ip
	return error,w,b+error
#Activation functions
reLU = jit(nopython=True,cache=True,nogil=True)(lambda n:n if 0<n else 0)
sigmoid  = jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda n:1/(1+(e**-n)))
softmax= jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda x:e**x)
tanh= jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda x:1-(2/(1+e**(2*x))))
actdic = {
	"sigmoid":lambda o:sigmoid(o),
	"relu": lambda o:reLU(o),
	"softmax": lambda o:softmax(o),
	"tanh": lambda o:tanh(o)
}
transfer_derivative = jit(nopython=True,fastmath=True,cache=True,nogil=True)(lambda output:output*(1-float(output)))
def save_model(Layers,name):
	zipfile = ZipFile(f"{name}.model","w")
	zipfile.writestr("ModelInfo",f"{len(Layers[0])},{len(Layers[1:-1])},{len(Layers[1])},{len(Layers[-1])},{Layers[1][0].activation}")
	for i,layer in enumerate(Layers,1):
		txt=""
		for neuron in layer:
			stxt = str(list(neuron.w))+str(neuron.b)+","+str(neuron.berror)+"\r\n"
			for j in "float(') [":
				stxt=stxt.replace(j,"")
			txt+=stxt
		zipfile.writestr(f"Layer{i}",txt)
	zipfile.close()
def regain_model(name):
	zipfile = ZipFile(f"{name}.model",'r')
	minfo=zipfile.read("ModelInfo").decode("utf-8").split(",")
	Layers = Network(*([int(i) for i in minfo[:-1]]+[False,minfo[-1]]))
	for i,layer in enumerate(Layers,1):
		file = zipfile.read(f"Layer{i}").decode("utf-8").splitlines()
		for l,neuron in zip(file,layer):
			l = l.split("]")
			neuron.w  = array([float(w) for w in l[0].split(",")])
			sp = l[1].split(",")
			neuron.b = float(sp[0])
			neuron.berror=float(sp[1])
	zipfile.close()
	return Layers
@jit(nopython=False,cache=True,fastmath=True,nogil=True)
def smax(l):
	m = max(l)
	l = [i-m for i in l]
	t = sum(l)
	return [i/t for i in l]
#Giving input
def ForwardPropagation(inp,Layers):
	threads=[]
	for neuron,i in zip(Layers[0],inp):
		p=Thread(target=neuron.input,args=(array([i]),))
		threads.append(p)
		p.start()
	for thread in threads:
		thread.join()
	#Forward Propagation
	for llayer,layer in  zip(Layers,Layers[1:]):
		ni = [float(n.o) for n in llayer]
		if llayer[0].activation=='softmax':
			ni = smax(ni)
		threads=[]
		ni=array(ni)
		for neuron in layer:
			p=Thread(target=neuron.input,args=(ni,))
			threads.append(p)
			p.start()
		for thread in threads:
			thread.join()
	return Layers
def fun(bl,i,neuron,learning_rate,momentum):
	neuron.error = 0
	for bneuron in bl:
		neuron.error+=bneuron.w[i]*bneuron.error
	neuron.error*=transfer_derivative(neuron.o)
	neuron.updateweight(learning_rate,momentum)
def funf(neuron,expected,learning_rate,momentum):
	output = neuron.o
	err = (expected-output)
	neuron.error=err*abs(err)*transfer_derivative(output)
	neuron.updateweight(learning_rate,momentum)
def bpropagation(out,Layers,learning_rate,momentum=0):
	#Backward Propagation
	Layers.reverse()
	threads=[]
	#c=float(1/len(Layers))
	for neuron,expected in zip(Layers[0],out):
		p = Thread(target=funf,args=(neuron,expected,learning_rate,momentum,))
		threads.append(p)
		p.start()
	for thread  in threads:
		thread.join()
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
def predict(inp,Layers):
	ForwardPropagation(inp,Layers)
	opl = [i.o for i in Layers[-1]]
	m = max(opl)
	return (opl.index(m),m,opl)
def printer(Layers):
	#Printing value of neurons
	for layer in Layers:
		print([str(n.o) for n in layer])
		print()
def wprinter(Layers):
	for layer in Layers:
		print([[list(map(str,i.w)),str(i.b)] for i in layer])
		print()
train  = lambda inp,out,learning_rate,Layers,momentum=0:bpropagation(out,ForwardPropagation(inp,Layers),float(learning_rate),momentum)
def traina(Layers,inp,out,learning_rate,momentum=0,iteration=1,pdisplay=False,clr=True):
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
			lr-=k
			bar.set_postfix_str(str(round(lr,3)))
	return Layers
#sn-No of starting neuron
#hl-No of hidden layers
#hn-No of neuron in hidden layers
#on-No of output neuron
def Network(sn,hl,hn,on,new=True,activation="reLU"):
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
	try:
		Layers = regain_model(model_name)
	except FileNotFoundError:
		print("Making",model_name)
		Layers  = Network(sn,hl,hn,on,activation=activation)
		save_model(Layers, model_name)
	return Layers
def check_err(Layers,ipa,opa):
	err=0
	ol=len(opa)
	for inp,out in zip(ipa,opa):
		for i in range(ol):
			err+=abs(ForwardPropagation(inp,Layers)[-1][i].o-out[i])
	return err