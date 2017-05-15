import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time

start_time = time.time()

#initialization
FILENAME = 'ratings_train.txt'
K = 20
#m = 10
#n = 25
max_iter = 40 # maximum no of iterations
lambda1 = 0.2 # regularization term
alpha = 0.03 #learning rate

def Sgrad(P,Q,R):
	E = 0
	for i in range(len(R)):
		for u in range(len(R[1])):
			#print np.dot(Q[i,:],transpose(P[u,:]))
			E = E + (R[i,u] - np.dot(Q[i,:],np.transpose(P[u,:])))**2 + regularization_term(Q[i,:],P[u,:])
			Eui = R[i,u] - np.dot(Q[i,:],np.transpose(P[u,:]))
			GDq = Q[i,:] + alpha*(np.dot(Eui,P[u,:]) - np.dot(lambda1,Q[i,:]))
			Q[i,:] = GDq
			GDp = P[u,:] + alpha*(np.dot(Eui,Q[i,:]) - np.dot(lambda1,P[u,:]))
			P[u,:] = GDp
	return E,P,Q

def regularization_term(P,Q):
	sum1 = np.sum(P) + np.sum(Q)
	return float(lambda1) * sum1

def initial(m,n):
	high = 5/math.sqrt(K)
	P = np.random.uniform(low=0.0, high=high, size=(n,K))
	Q = np.random.uniform(low=0.0, high=high, size=(m,K))
	P = np.asmatrix(P)
	Q = np.asmatrix(Q) # if you want to conver to np.matrix 
	return P,Q

def main():
	array = []
	file = open(FILENAME,'r').readlines()
	for item in file:
		item = item.split('\t')
		x = [int(item[0]),int(item[1]),int(item[2])]
		array.append(x)
	array = np.asarray(array)

	m = (max(array[:,0]))
	n = (max(array[:,1]))
	#FileR = np.zeros((943,1682))
	FileR = np.zeros((m,n))
	count = 0
	for x in array:
		print 'reading the file'
		if count % 10000 == 0:
			print 'progress %d/%d' % (1000+count,len(array))
		FileR[x[0]-1][x[1]-1] = x[2]-1
		count = count + 1
	R = FileR
	#P,Q,R = initial(m,n,K) 
	high = 5/math.sqrt(K)
	P = np.random.uniform(low=0.0, high=high, size=(n,K))
	Q = np.random.uniform(low=0.0, high=high, size=(m,K))
	#shuffling the matrices, P & Q are random already
	P = np.random.shuffle(P)
	Q = np.random.shuffle(Q)
	R = np.random.shuffle(R)
	Err = []
	for it in range(max_iter):
		try:
			print 'itearation  %d/%d' % (it+1,max_iter)
			E,P,Q = Sgrad(P,Q,R)
			if len(Err) > 1:
				#if the error is zero
				if Err[-1] == E:
					break
			Err.append(float(E))
		except:
			pass
	print Err
	plt.plot(range(max_iter), Err)
	plt.title('K = %d, Lambda = %.2f, Learning rate = %.2f' % (K,lambda1,alpha))
	#plt.text(max_iter-10, Err[1], r'$\lambda=100,\ \alpha=0.03$')
	plt.ylabel('Error')
	plt.xlabel('iterations')
	print("--- %s seconds ---" % (time.time() - start_time))
	plt.show()

if __name__ == "__main__":
	main()
