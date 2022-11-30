import numpy as np
import random
import pickle
import sys

def generateLDA(M, N, T, V, alpha, beta):
	alpha = [alpha]*T
	beta = [beta]*V
	words = []
	documents = []
	all_topics=[]
	doc_distributions = []
	word_distributions = {}

	for i in range(V):
		word_distributions.update({i:[0]*T})

	word_probs = np.random.dirichlet(beta, T)

	for i in range(M):
		topics=[]
		doc_distribution = [0]*T
		if i%10==0:
			print(i)
		topic_probs = np.random.dirichlet(alpha, 1)
		topic_probs = topic_probs.T
		N_i = random.randint(1,N)
		for j in range(N_i):
			topic = np.random.multinomial(1,np.squeeze(topic_probs))
			word = np.random.multinomial(1, word_probs[list(topic).index(1)])
			all_topics.append(topic)
			topics.append(topic)
			words.append(list(word).index(1))
		documents.append([words, topics])

		for t in topics:
			doc_distribution = np.asarray(t) + np.asarray(doc_distribution)
		doc_distributions.append(doc_distribution)

	for word in range(V):
		for j in range(len(words)):
			if words[j]==word:
				word_distributions[word]=word_distributions[word] + all_topics[j]

	return documents, word_distributions, doc_distributions


#C1, word_distributions, doc_distributions = generateLDA(10000,100,10,50000,0.1,0.01)
C2, word_distributions, doc_distributions = generateLDA(100000,100,100,50000,0.1,0.01)
#C3 = generateLDA(100000,10,1000,50000,0.1,0.01)



print("Saving...")
file = open('C1', 'wb')
pickle.dump([C2, word_distributions, doc_distributions], file)
file.close()
