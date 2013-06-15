from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import random


class SentimentClassifier(object):
	"""
	Uses Sentiment Polarity Datasert Version 2.0. It's pretty good but it's based only on
	movie review and a lot of movie specific terms tend to show up as most informative
	features. A dataset based on more diverse sources might produce better results.
	"""

	def __init__(self):
		documents = [(list(movie_reviews.words(fileid)), category)
		              for category in movie_reviews.categories()
		              for fileid in movie_reviews.fileids(category)]
		random.shuffle(documents)

		all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
		self.__word_features = set(all_words.keys()[:2000])

		featuresets = [(self.__document_features(d), c) for (d,c) in documents]
		train_set, test_set = featuresets[500:], featuresets[:500]
		self.__base_clasifier = nltk.NaiveBayesClassifier.train(train_set)
		# print nltk.classify.accuracy(self.__base_clasifier, test_set)


	def __document_features(self, document): 
	    features = {}
	    for word in document:
	    	if word in self.__word_features:
   				features['contains(%s)' % word] = True   
	    return features


	def classify(self, document):
		"""
		Classifies a document as positive or negative.
		Will return a pair of tupples with the format '(categoryName, probability)'
		"""
		words = [word for sent in sent_tokenize(document) for word in word_tokenize(sent)]
		probDist = self.__base_clasifier.prob_classify(self.__document_features(words))
		return [(sample, probDist.prob(sample)) for sample in probDist.samples()]
