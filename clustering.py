import matplotlib.pyplot as plt
import pickle
import random

# Question 1
def plot(collection):
	for i in range(1):
		document = random.choice(collection)
		print("Document", document)
		print("Length of T = ", len(document))
		word = random.choice(document)
		print("Word", word)
		plt.hist(document, bins=len(document))
		plt.show()

 # Q 1 (5 marks). For each of the 3 collections,
 # sample 5 documents and 5 words at random and plot each
 # T -dimensional vector (the distributions) as a bar chart.

collections = []
print("Loading collections...")
for c in ['C1', 'C2', 'C3']:
	file = open(c, 'rb')
	collections.append(pickle.load(file))
	file.close()


# Question 3
def cosineSimilarity(x, y):
	sum=0
	for i in range(len(x)):
		sum+=x[i]*y[i]
	return sum/(len(x)*len(y))

def averageSimilarity(collection):
	print(len(collection))
	sim=0
	pairs = []
	for i in range(100):
		print(i)
		pairs.append([random.choice(collection), random.choice(collection)])

	for pair in pairs:
		sim += cosineSimilarity(pair[0], pair[1])
	return sim/100

print("Average document similarity of C1:", averageSimilarity(collections[0]))
print("Average document similarity of C2:", averageSimilarity(collections[1]))
print("Average document similarity of C3:", averageSimilarity(collections[2]))
