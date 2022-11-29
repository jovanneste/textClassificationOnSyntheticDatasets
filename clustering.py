import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from sklearn.cluster import KMeans

# Question 1
def plot(collection):
	topic_num = (len(collection[0][1][0]))
	chosen_topics = []
	list_sum = [0]*topic_num
	for i in range(5):
		document = random.choice(collection)
		topics = document[1]
		for j in range(5):
			index = random.choice(range(len(topics)))
			chosen_topics.append(topics[index])

	for t in chosen_topics:
		list_sum = np.asarray(t) + np.asarray(list_sum)


	plt.bar(list(range(1, topic_num+1)), list_sum)
	plt.title('C1 distribution')
	plt.xlabel('Topics')
	plt.ylabel('Occurences')
	plt.show()

collections = []
print("Loading collections...")

file = open('C1', 'rb')
C1, word_distributions, doc_distributions = pickle.load(file)
file.close()


# Question 3
def cosineSimilarity(x, y):
	sum=0
	for i in range(len(x)):
		sum+=x[i]*y[i]
	return sum/(len(x)*len(y))

def averageSimilarity(collection):
	sim=0
	pairs = []
	for i in range(100):
		pairs.append([random.choice(collection), random.choice(collection)])

	for pair in pairs:
		sim += cosineSimilarity(pair[0], pair[1])
	return sim/100

# Question 4

def cluster(words, k):
	try:
		vectors = list(words.values())
	except:
		vectors = words
	vectors = np.array(vectors)
	print(vectors.shape)
	print("Clustering...")
	kmeans = KMeans(n_clusters=k, random_state=0).fit(vectors)
	clusters = kmeans.labels_
	print(len(clusters))
	print(clusters)


# print("Average document cosine similarity:", averageSimilarity(doc_distributions))
# print("Average word cosine similarity:", averageSimilarity(word_distributions))

print(cluster(doc_distributions, 20))
