import matplotlib.pyplot as plt
import pickle
import random

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

file = open('C1', 'rb')
C1 = pickle.load(file)
file.close()

