import numpy as np


def generateLDA(M, N, T, V, alpha, beta):
	alpha = [alpha]*T
	beta = [beta]*V
	words = []
	
	word_probs = np.random.dirichlet(beta, T)

	for i in range(M):
		topic_probs = np.random.dirichlet(alpha, 1)
		topic_probs = topic_probs.T
		N_i = 10#random.randint(1,N)
		for j in range(N_i):
			topic = np.random.multinomial(1,np.squeeze(topic_probs))
			word = np.random.multinomial(1, word_probs[list(topic).index(1)])
			words.append(list(word).index(1))
            
	return words


print(generateLDA(1, 10, 5, 3, 0.5, 0.3))
