import numpy as np
import random
import pickle 

def generateLDA(M, N, T, V, alpha, beta):
	alpha = [alpha]*T
	beta = [beta]*V
	words = []
	documents = []
	
	word_probs = np.random.dirichlet(beta, T)

	for i in range(M):
		if i%100==0:
			print(i)
		topic_probs = np.random.dirichlet(alpha, 1)
		topic_probs = topic_probs.T
		N_i = random.randint(1,N)
		for j in range(N_i):
			topic = np.random.multinomial(1,np.squeeze(topic_probs))
			word = np.random.multinomial(1, word_probs[list(topic).index(1)])
			words.append(list(word).index(1))
		documents.append(words)
            
	return documents


C1 = generateLDA(10000,100,10,50000,0.1,0.01)
# C2 = generateLDA(100000,100,100,50000,0.1,0.01)  
# C3 = generateLDA(100000,10,1000,50000,0.1,0.01)

file = open('C1', 'wb')
pickle.dump(C1, file)
file.close()




