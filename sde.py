import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
#dy/dx = -ky


class SDESolve:
	def __init__(self,f=lambda x: 1.0, h=lambda x: 0.0):
		self.f = f
		self.h = h
		
	
	def simulate(self,x_init,t_max,t_init=0.0,n_steps=10000,method='forward',n_paths=100,save_n=10):
		
		
		np.random.seed(0)
		x_vec = np.zeros(n_paths)+x_init
		
		x_store=np.zeros((n_paths,int(n_steps/save_n)+1))
		t_store=np.zeros(int(n_steps/save_n)+1)
		
		x_store[:,0] += x_init
		t_store[0]= t_init
		kk=1
		
		
		
		dt = (t_max-t_init)/(n_steps-1)
		
		noise_old = np.random.randn(n_paths)
		for i in range(0,n_steps):
			noise_vec = np.random.randn(n_paths)/np.sqrt(dt)
			
			t=t_init+i*dt
			
			if method=="forward":
				x_vec = self.forward_step(x_vec,t,dt,self.f,self.h,noise_vec)
			elif method == "backward":
				x_vec = self.backward_step(x_vec,t,dt,self.f,self.h,noise_vec)
			elif method == "midpoint":
				x_vec = self.midpoint_step(x_vec,t,dt,self.f,self.h,noise_vec)
			elif method == "CN":
				x_vec = 0.5*self.forward_step(x_vec,t,dt,self.f,self.h,noise_vec)+0.5*self.backward_step(x_vec,t,dt,self.f,self.h,noise_vec)
				
				
			if (i+1)%save_n == 0:
				t_store[kk] = t
				x_store[:,kk] = x_vec
				kk+=1
			noise_old = np.copy(noise_vec)
		return x_store,t_store
	def backward_step(self,x_0,t,dt,f,h,noise):
		
		def errf(x,args):
			[x_0,t,dt,f,h,noise]=args
			return x-x_0-dt*(f(x,t+dt)+h(x,t+dt)*noise)

		
		x = np.zeros_like(x_0)
		for i in range(0,len(x_0)):
			res = root(errf,np.array([x_0[i]]),args=[x_0[i],t,dt,f,h,noise[i]],tol=1e-10)
			#assert(res["success"])
			x[i]=res['x'][0]
		return x
	def forward_step(self,x_0,t,dt,f,h,noise):
		return x_0+dt*(f(x_0,t)+h(x_0,t)*noise)
		
		
	def midpoint_step(self,x_0,t,dt,f,h,noise):
		def errf(x,args):
			[x_0,t,dt,f,h,noise]=args
			return x-x_0-dt*(f(0.5*(x_0+x),t+0.5*dt)+h(0.5*(x_0+x),t+0.5*dt)*noise)

		x = np.zeros_like(x_0)
		for i in range(0,len(x_0)):
			res = root(errf,np.array([x_0[i]]),args=[x_0[i],t,dt,f,h,noise[i]],tol=1e-10)
			#assert(res["success"])
			x[i]=res['x'][0]
		return x
		
		
def f(x,t):
	return np.sin(5*t)*x
def h(x,t):
	return 1.0

	
methods=["forward","backward","midpoint","CN"]
for meth in methods:
	ss=SDESolve(f,h)
	x,t=ss.simulate(1.0,3.0,n_steps=1000,save_n=1,n_paths=1,method=meth)

	mu=np.mean(x,0)
	std=np.std(x,0)
	plt.plot(t,mu)

plt.legend(methods)
plt.show()