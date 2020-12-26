import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import tflearn
import json
import random
import pickle

class ChatBot():
	def __init__(self, file="intents.json"):
		self.stemmer	= LancasterStemmer()
		try:
			with open("data.pickle", "rb") as f:
				self.words, self.labels, self.training, self.output = pickle.load(f)
		except:
			self._preprocessing(file)
			self._data_preprocessing()
		self._model()

	def _preprocessing(self, file):
		words	= []
		labels	= []
		docs_x	= []
		docs_y	= []

		with open("intents.json") as file:
			data = json.load(file)

		for intent in data["intents"]:
			for pattern in intent["patterns"]:
				wrds = nltk.word_tokenize(pattern)
				words.extend(wrds)
				docs_x.append(wrds)
				docs_y.append(intent["tag"])

				if intent["tag"] not in labels:
					labels.append(intent["tag"])

		words		= [self.stemmer.stem(w.lower()) for w in words if w not in "?"]
		words		= sorted(list(set(words)))
		labels		= sorted(labels)
		self.words	= words
		self.labels	= labels
		self.docs_x	= docs_x
		self.docs_y	= docs_y

	def _data_preprocessing(self):
		training	= []
		output		= []
		out_empty	= [0 for _ in range(len(self.labels))]

		for x, doc in enumerate(self.docs_x):
			bag		= []
			wrds	= [self.stemmer.stem(w) for w in doc]

			for w in self.words:
				if w in wrds:
					bag.append(1)
				else:
					bag.append(0)
			output_row = out_empty[:]
			output_row[self.labels.index(self.docs_y[x])] = 1
			training.append(bag)
			output.append(output_row)

		self.training	= np.array(training)
		self.output		= np.array(output)
		with open("data.pickle", "wb") as f:
			pickle.dump((self.words, self.labels, self.training, self.output), f)

	def _model(self):
		tf.compat.v1.get_default_graph()
		net		= tflearn.input_data(shape=[None, len(self.training[0])])
		net		= tflearn.fully_connected(net, 8)
		net		= tflearn.fully_connected(net, 8)
		net		= tflearn.fully_connected(net, len(self.output[0]), activation="softmax")
		net		= tflearn.regression(net)
		model	= tflearn.DNN(net)

		model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
		model.save("model.tflearn")
		self.model = model

	def _bag_of_words(self, s):
		bag		= [0 for _ in range(len(self.words))]
		s_words	= nltk.word_tokenize(s)
		s_words	= [self.stemmer.stem(word.lower()) for word in s_words]

		for se in s_words:
			for i, w in enumerate(self.words):
				if w == se:
					bag[i] = 1
		return bag

	def chat(self):
		print("Start talking with the bot!")
		while True:
			inp	= input("You: ")
			if inp.lower() == "quit":
				break

			results = self.model.predict([self._bag_of_words(inp)])
			results_index = np.argmax(results)
			tag = self.labels[results_index]

			with open("intents.json") as file:
				data = json.load(file)

			for tg in data["intents"]:
				if tg["tag"] == tag:
					responses = tg["responses"]

			print(random.choice(responses))

def main():
	Bot	= ChatBot()
	Bot.chat()

if __name__ == "__main__":
	main()