from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy
from scipy.stats import cmedian, tmean
from collections import defaultdict
import datetime

from pos_vectorize import loadData
from sentimentAnalysis import SentimentClassifier


__all__ = ['ReviewLengthEstimator', 'UnigramEstimator', 'UserReviewCountEstimator', 
	'SentenceCountEstimator', 'AverageSentenceLengthEstimator', 'ParagraphCountEstimator',
	'POSPipleline', 'SentimentEstimator', 'BusinessReviewCountEstimator', 'WinnerBiasEstimator',
	'ReviewAgeEstimator', 'AverageParagraphLength', 'CheckinsCountEstimator', 
	'BusinessCategoriesEstimator']


class ReviewLengthEstimator(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			text_length = float(len(review['text']))
			feature_matrix.append([text_length, text_length**2])
		return feature_matrix


class ReviewTextTransformer(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		review_texts = []
		for review in X:
			review_texts.append(review['text'])
		return review_texts


class UnigramEstimator(Pipeline):

	def __init__(self):
		super(UnigramEstimator, self).__init__([('review_text', ReviewTextTransformer()),
												('hash_vect', HashingVectorizer()), 
												('tfidf', TfidfTransformer()),
												])
		# frequency normalization will be done in tfidf step anyway
		self.set_params(hash_vect__norm=None, hash_vect__non_negative=True)


class UserReviewCountEstimator(BaseEstimator):

	def __init__(self, data):
		super(UserReviewCountEstimator, self).__init__()
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			user_id = review['user_id']
			user_review_count = 0.0 #this may have to be set to a mean value
			#not all reviews have a profile
			if user_id in self.data.users:
				user = self.data.users[user_id]
				user_review_count = float(user['review_count'])
			feature_matrix.append([user_review_count, user_review_count**2])
		return feature_matrix


class SentenceCountEstimator(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			sentence_count = len(sent_tokenize(review['text']))
			feature_matrix.append([sentence_count])
		return feature_matrix


class AverageSentenceLengthEstimator(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			words_count = 0
			sentence_count = 0
			for sentence in sent_tokenize(review['text']):
				words_count += len(word_tokenize(sentence))
				sentence_count += 1
			average_length = words_count / float(sentence_count)
			feature_matrix.append([average_length, average_length**2])
		return feature_matrix


class ParagraphCountEstimator(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			paragraphs = float(len(review['text'].splitlines()))
			feature_matrix.append([paragraphs])
		return feature_matrix


class AverageParagraphLength(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			paragraph_count = len(review['text'].splitlines())
			if paragraph_count > 0:
				average_length = float(len(review['text']) / float(paragraph_count))
			else:
				average_length = 0
			feature_matrix.append([average_length, average_length**2])
		return feature_matrix


class POSEstimator(BaseEstimator):

	def __init__(self, pos_data=None):
		self.pos_data = pos_data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			review_id = review['review_id']
			feature_list = self.pos_data['reviews'][review_id]
			feature_matrix.append(feature_list)
		return feature_matrix


class POSSelector(BaseEstimator):

	def __init__(self, tag_list=None, pos_data=None):
		self.tag_list = tag_list
		self.pos_data = pos_data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for tag_name, tag_index in self.pos_data['tag_indexes'].iteritems():
			if tag_name in self.tag_list:
				column = X.getcol(tag_index).toarray()
				if len(feature_matrix) == 0:
					feature_matrix = column
				else:
					feature_matrix = numpy.hstack((feature_matrix, column))
				feature_matrix = numpy.hstack((feature_matrix, column**2))
		return feature_matrix


class POSPipleline(Pipeline):
	"""
	The most promising tags were:
	',', 'CC', 'CD', 'DT', 'IN', 'MD', 'NNS', 'PRP', 'PRP$', 'RP', 'TO', 'VB', 'VBD', 'VBG', 'WRB'
	"""

	__pos_data = None

	@classmethod
	def pos_data(cls):
		if cls.__pos_data is None:
			cls.__pos_data = loadData()
		return cls.__pos_data


	def __init__(self):
		super(POSPipleline, self).__init__([('pos', POSEstimator()), 
											('tfidf', TfidfTransformer()),
											('pos_select', POSSelector())])
		self.set_params(pos__pos_data=POSPipleline.pos_data(), 
						pos_select__pos_data=POSPipleline.pos_data())


class SentimentEstimator(BaseEstimator):

	__classifier = None

	@classmethod
	def classifier(cls):
		if cls.__classifier is None:
			cls.__classifier = SentimentClassifier()
		return cls.__classifier

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			classification = self.classifier().classify(review['text'])
			score = -classification[0][1] + classification[1][1]
			row = [score, score**2]
			feature_matrix.append(row)
		return feature_matrix


class BusinessReviewCountEstimator(BaseEstimator):

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			business = self.data.get_business_for_review(review)
			count = float(business['review_count'])
			feature_matrix.append([count])
		return feature_matrix


class WinnerBiasEstimator(BaseEstimator):

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		self.business_winner_bias = {}
		business_review_votes = defaultdict(list)
		for review in self.data.training_reviews.values():
			business_review_votes[review['business_id']].append(review['votes']['useful'])
		for business_id, review_votes in business_review_votes.iteritems():
			median = cmedian(review_votes)
			mean = tmean(review_votes)
			if len(review_votes) > 0 and mean != 0:
				bias = median / mean
			else:
				bias = 1
			self.business_winner_bias[business_id] = bias

		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			business_id = self.data.get_business_for_review(review)['business_id']
			bias = self.business_winner_bias[business_id]
			feature_matrix.append([bias, bias**2, bias**3])
		return feature_matrix


class ReviewAgeEstimator(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			date_format = '%Y-%m-%d'
			draft_date = datetime.datetime.strptime(review['date'], date_format)
			snapshot_date = datetime.datetime.strptime(review['snapshot_date'], date_format)
			time_delta = float((snapshot_date - draft_date).days)
			feature_matrix.append([time_delta])
		return feature_matrix


class CheckinsCountEstimator(BaseEstimator):

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			business = self.data.get_business_for_review(review)
			business_id = business['business_id']
			checkins_count = 0
			if business_id in self.data.training_checkins:
				checkins = self.data.training_checkins[business_id]
				checkins_count = float(sum(checkins['checkin_info'].values()))
			feature_matrix.append([checkins_count])
		return feature_matrix


class BusinessCategoriesEstimator(BaseEstimator):
	"""
	WARNING
	Works only with a modified version of LabelBinarizer
	"""

	def __init__(self, data=None):
		self.data = data

	def __create_labels_list(self, review_list):
		labels = []
		for review in review_list:
			business = self.data.get_business_for_review(review)
			labels.append(business['categories'])
		return labels

	def fit(self, X, y):
		self.binarizer = LabelBinarizer()
		labels = self.__create_labels_list(X)
		self.binarizer.fit(labels)
		return self

	def transform(self, X):
		labels = self.__create_labels_list(X)
		binarized_labels = self.binarizer.transform(labels)
		return binarized_labels.astype(float)
