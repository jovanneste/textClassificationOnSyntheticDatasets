import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from sklearn.cluster import KMeans
import sys

# from topics of a document return the topic distribution of that document
def documentTopic(topics):
	list_sum = [0]*len(topics[0])
	for t in topics:
		list_sum = np.asarray(t) + np.asarray(list_sum)
	return list_sum

# get topic distribution for jth word
def topicWord(word_topics, j):
	return word_topics[j]

# Question 1
def plot(collection):
	topic_num = (len(collection[0][1][0]))
	chosen_topics = []
	for i in range(5):
		document = random.choice(collection)
		topics = document[1]
		for j in range(5):
			index = random.choice(range(len(topics)))
			chosen_topics.append(topics[index])
	document_topic_dist = documentTopic(chosen_topics)
	plt.bar(list(range(1, topic_num+1)), document_topic_dist)
	plt.xlabel('Topics')
	plt.ylabel('Occurences')
	plt.show()


# Question 3
def cosineSimilarity(x, y):
	sum = np.dot(x, y)
	sum_of_squares = float(np.linalg.norm(x)*np.linalg.norm(y))
	return sum/sum_of_squares

def averageSimilarity(collection):
	if type(collection)!=list:
		collection = list(collection.values())
	sim=0
	pairs = []
	for i in range(100):
		pairs.append([random.choice(collection), random.choice(collection)])

	for pair in pairs:
		sim += cosineSimilarity(pair[0], pair[1])
	return sim/100


# Question 4
def purity(clusters, classes, k):
	max_topics = 0
	n=0
	for i in range(k):
		intersection = []
		for j in clusters[i][0]:
			n+=1
			intersection.append(classes[j])
		max_topics += intersection.count(max(intersection))
	return max_topics/n


def cluster(words, k):
	try:
		vectors = list(words.values())
	except:
		vectors = words
	vectors = np.array(vectors)
	print("Clustering...")
	kmeans = KMeans(n_clusters=k, random_state=0).fit(vectors)
	clusters = kmeans.labels_
	sigma = []
	classes = []
	for i in range(k):
		sigma.append(np.where(clusters==i))

	for document in list(words):
		classes.append(np.argmax(document))

	print("Purity with " + str(k) + " clusters: " + str(purity(sigma, classes, k)))


if __name__ == '__main__':
	file = open('C1', 'rb')
	C1, word_distributions, doc_distributions = pickle.load(file)
	file.close()

	print(topicWord(word_distributions, 5))

	plot(C1)

	# remove all words that do not show up in document
	word_distributions = {x:y for x,y in word_distributions.items() if list(y)!=[0]*len(word_distributions[0])}
	print("Average document cosine similarity:", averageSimilarity(doc_distributions))
	print("Average word cosine similarity:", averageSimilarity(word_distributions))

	Ks = [20, 100, 500, 5000]
	for k in Ks:
		cluster(doc_distributions, k)
