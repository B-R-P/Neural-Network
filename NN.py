from random import randrange
from decimal import Decimal
from zipfile import ZipFile
from math import e,tanh
from numpy import dot,exp,sum
from tqdm import tqdm
from threading import Thread
e=Decimal(e)
class Neuron():
	count = 0
	def __init__(self,inum,new,activation):
		if new:
			Neuron.count+=1
			self.b = sigmoid(self.count)
			if .5<self.b:
				self.b = 1-self.b
			self.w = [1/Decimal(randrange(1,8)) for i in range(1,inum+1)]
			self.berror=0
		self.activation  = activation
	def input(self,inp):
		self.inp = inp
		self.o = dot(self.w,inp)+self.b
		if self.activation=="sigmoid":
			self.o = sigmoid(self.o)
		elif self.activation=="reLU":
			self.o = reLU(self.o)
		elif self.activation=="softmax":
			self.o = softmax(self.o)
		elif self.activation=="tanh":
			self.o=Decimal(tanh(self.o))
	def updateweight(self,learning_rate,momentum):
		self.error*=learning_rate
		self.error+=(self.berror*momentum)
		for w,ip in enumerate(self.inp):
			self.w[w]+=self.error*ip
		self.b+=self.error
		self.berror=self.error
	def output(self):
		return self.o
def reLU(n):
	return Decimal(max(0,n))
def sigmoid(n):
	return 1/(1+(e**-Decimal(n)))
def softmax(x):
	return x/sum(exp(x),axis=0)
def transfer_derivative(output):
	return output*(1-Decimal(output))
def save_model(Layers,name):
	zipfile = ZipFile(f"{name}.model","w")
	zipfile.writestr("ModelInfo",f"{len(Layers[0])},{len(Layers[1:-1])},{len(Layers[1])},{len(Layers[-1])},{Layers[1][0].activation}")
	for i,layer in enumerate(Layers,1):
		txt=""
		for neuron in layer:
			stxt = str(neuron.w)+str(neuron.b)+","+str(neuron.berror)+"\r\n"
			for j in "Decimal(') [":
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
			neuron.w  = [Decimal(w) for w in l[0].split(",")]
			sp = l[1].split(",")
			neuron.b = Decimal(sp[0])
			neuron.berror=Decimal(sp[1])
	zipfile.close()
	return Layers
#Giving input
def ForwardPropagation(inp,Layers):
	threads=[]
	for neuron,i in zip(Layers[0],inp):
		p=Thread(target=neuron.input,args=([Decimal(i)],))
		threads.append(p)
		p.start()
	for thread in threads:
		thread.join()
	#Forward Propagation
	for llayer,layer in  zip(Layers,Layers[1:]):
		ni = [Decimal(n.output()) for n in llayer]
		threads=[]
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
	neuron.error*=transfer_derivative(neuron.output())
	neuron.updateweight(learning_rate,momentum)
def funf(neuron,expected,learning_rate,momentum):
	output = Decimal(neuron.output())
	neuron.error=(expected-output)*transfer_derivative(output)
	neuron.updateweight(learning_rate,momentum)
def bpropagation(out,Layers,learning_rate,momentum):
	#Backward Propagation
	Layers.reverse()
	threads=[]
	#c=Decimal(1/len(Layers))
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
	opl = [i.output() for i in Layers[-1]]
	m = max(opl)
	return (opl.index(m),m,opl)
def printer(Layers):
	#Printing value of neurons
	for layer in Layers:
		print([str(n.output()) for n in layer])
def train(inp,out,learning_rate,momentum,Layers):
	return bpropagation(out,ForwardPropagation(inp,Layers),Decimal(learning_rate),momentum)
def traina(Layers,inp,out,learning_rate,momentum=0,iteration=1,pdisplay=True,clr=True):
	lr = Decimal(learning_rate)
	momentum=Decimal(momentum)
	k=lr/iteration
	c = list(zip(inp,out))
	bar=tqdm(range(iteration))
	for j in bar:
		if pdisplay:
			if clr:
				print("Iteration:",j+1,"Learning Rate:",lr)
			else:
				print("Iteration:",j+1)
		for ip,op in c:
			Layers=bpropagation(op,ForwardPropagation(ip,Layers),lr,momentum=momentum)
		if clr:
			lr-=k
			bar.set_postfix_str(str(round(lr,5)))
	return Layers
#sn-No of starting neuron
#hl-No of hidden layers
#hn-No of neuron in hidden layers
#on-No of output neuron
def Network(sn,hl,hn,on,new=True,activation):
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
		Layers[-1]+=[Neuron(hn,new,activation="tanh")]
	return Layers
def model(sn,hl,hn,on,model_name,activation="reLU"):
	try:
		Layers = regain_model(model_name)
	except Exception as e:
		print("Making",model_name)
		Layers  = Network(sn,hl,hn,on,activation)
	return Layers
def check_err(Layers,ipa,opa):
	err=0
	ol=len(opa)
	for inp,out in zip(ipa,opa):
		for i in range(ol):
			err+=abs(ForwardPropagation(inp,Layers)[-1][i].output()-out[i])
	return err