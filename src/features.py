from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy
from scipy.stats import cmedian, tmean
from collections import defaultdict, Counter

from pos_vectorize import loadData
from sentimentAnalysis import SentimentClassifier


__all__ = ['ReviewLengthFeature', 'UnigramFeature', 'UserReviewCountFeature', 
	'SentenceCountFeature', 'AverageSentenceLengthFeature', 'ParagraphCountFeature',
	'POSFeature', 'SentimentFeature', 'BusinessReviewCountFeature', 'WinnerBiasFeature',
	'ReviewAgeFeature', 'AverageParagraphLengthFeature', 'CheckinsCountFeature', 
	'BusinessCategoriesFeature', 'TimeCompetitionFeature', 'UserUsefulVotesFeature',
	'UserFunnyVotesFeature', 'UserCoolVotesFeature', 'PunctuationFeature',
	'BusinessOpenFeature', 'UserStarsDistributionFeature']


class ReviewLengthFeature(BaseEstimator):
	"""
	The length in characters in each review.
	"""

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


class UnigramFeature(Pipeline):
	"""
	Constructs a feature matrix where each column represents a separate unigram. The values
	are the tfidfs of the unigrams in the context of each review.
	"""

	def __init__(self):
		super(UnigramFeature, self).__init__([('review_text', ReviewTextTransformer()),
												('hash_vect', HashingVectorizer()), 
												('tfidf', TfidfTransformer()),
												])
		# frequency normalization will be done in tfidf step anyway
		self.set_params(hash_vect__norm=None, hash_vect__non_negative=True)


class UserReviewCountFeature(BaseEstimator):
	"""
	The number of reviews written by each review's user
	"""

	def __init__(self, data):
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


class SentenceCountFeature(BaseEstimator):
	"""
	The number of sentences in each review.
	"""

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			sentence_count = len(sent_tokenize(review['text']))
			feature_matrix.append([sentence_count])
		return feature_matrix


class AverageSentenceLengthFeature(BaseEstimator):
	"""
	The average length of sentences in each reviews. Sentence length is given by
	character count.
	"""

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


class ParagraphCountFeature(BaseEstimator):
	"""
	The number of paragraphs in each review.
	"""

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			# TODO: make this not count successive line splits
			paragraphs = float(len(review['text'].splitlines()))
			feature_matrix.append([paragraphs])
		return feature_matrix


class AverageParagraphLengthFeature(BaseEstimator):
	"""
	Average paragraph length, in characters, of each review.
	"""

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			# TODO: make this not count successive line splits
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


class POSFeature(Pipeline):
	"""
	This feature constructs a matrix where items represent the relative frequency of review parts
	of speech vs dataset parts of speech.
	"""

	__pos_data = None

	@classmethod
	def pos_data(cls):
		if cls.__pos_data is None:
			cls.__pos_data = loadData()
		return cls.__pos_data


	def __init__(self):
		super(POSFeature, self).__init__([('pos', POSEstimator()), 
											('tfidf', TfidfTransformer()),
											('pos_select', POSSelector())])
		self.set_params(pos__pos_data=POSFeature.pos_data(), 
						pos_select__pos_data=POSFeature.pos_data())


class SentimentFeature(BaseEstimator):
	"""
	This feature characterizes the relative sentiment of a review. See SentimentClassifier.
	"""

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


class BusinessReviewCountFeature(BaseEstimator):
	"""
	The number of reviews each review's business has received.
	"""

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


class WinnerBiasFeature(BaseEstimator):
	"""
	This feature attempts to give a per-business score that characterizes the imbalance in number
	of votes between the upper crust of best rated reviews vs the rest.
	"""

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
			if business_id in self.business_winner_bias:
				bias = self.business_winner_bias[business_id]
			else:
				bias = 1
			feature_matrix.append([bias, bias**2, bias**3])
		return feature_matrix


class ReviewAgeFeature(BaseEstimator):
	"""
	The age of the review in relation to the given snapshot dates. The snapshot dates were injected
	on data read. The review date is deliberatly taken as a reliable indicator of the review's age,
	ignoring the fact that the review may not have been publicly visible from that moment.
	"""

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			draft_date = review['date']
			snapshot_date = review['snapshot_date']
			time_delta = float((snapshot_date - draft_date).days)
			feature_matrix.append([time_delta])
		return feature_matrix


class CheckinsCountFeature(BaseEstimator):
	"""
	A per business feature that counts the number of checkins.
	"""

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


class BusinessCategoriesFeature(BaseEstimator):
	"""
	WARNING!!!
	Works only with a modified version of LabelBinarizer.

	A binarization of the reviews' business categories.
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


class TimeCompetitionFeature(BaseEstimator):
	"""
	This feature represents the time distance between each review and the closest in time review
	for the same business.
	"""

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def __reviews_indexed_by_business(self):
		reviews_dict = defaultdict(list)
		all_reviews = self.data.training_reviews.values() + self.data.test_reviews.values()
		for review in all_reviews:
			business_id = review['business_id']
			reviews_dict[business_id].append(review)
		return reviews_dict

	def transform(self, X):
		feature_matrix = []
		business_indexed_reviews = self.__reviews_indexed_by_business()
		for review in X:
			business = self.data.get_business_for_review(review)
			business_reviews = business_indexed_reviews[business['business_id']]

			dates = [r['date'] for r in business_reviews if r['review_id'] != review['review_id']]
			if len(dates) > 0:
				distance = float(min([(date - review['date']).days for date in dates]))
			else:
				distance = 0

			feature_matrix.append([distance, distance**2])
		return feature_matrix


class UserUsefulVotesFeature(BaseEstimator):
	"""
	The number of useful votes received by the review's user.
	"""

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			user_id = review['user_id']
			votes = 0.0
			if user_id in self.data.users:
				user = self.data.users[review['user_id']]
				if 'votes' in user:
					votes = float(user['votes']['useful'])
			feature_matrix.append([votes, votes**2, votes**3])
		return feature_matrix


class UserFunnyVotesFeature(BaseEstimator):
	"""
	The number of funny votes received by the review's user.
	"""

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			user_id = review['user_id']
			votes = 0.0
			if user_id in self.data.users:
				user = self.data.users[review['user_id']]
				if 'votes' in user:
					votes = float(user['votes']['funny'])
			feature_matrix.append([votes, votes**2])
		return feature_matrix


class UserCoolVotesFeature(BaseEstimator):
	"""
	The number of cool votes received by the review's user.
	"""

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			user_id = review['user_id']
			votes = 0.0
			if user_id in self.data.users:
				user = self.data.users[review['user_id']]
				if 'votes' in user:
					votes = float(user['votes']['cool'])
			feature_matrix.append([votes, votes**2])
		return feature_matrix


class PunctuationFeature(BaseEstimator):
	"""
	A template feature that counts the number of occurrences of the input punctuation sign 
	in a review, weighted by the review length.
	"""

	def __init__(self, sign=None):
		self.sign = sign

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			text = review['text']
			sign_count = text.count(self.sign)
			score = float(sign_count) / (len(text) + 1)
			feature_matrix.append([score])
		return feature_matrix


class BusinessOpenFeature(BaseEstimator):
	"""
	A binary feature that describes the review's business open status.
	"""

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			business = self.data.get_business_for_review(review)
			feature = float(business['open'])
			feature_matrix.append([feature])
		return feature_matrix


class UserStarsDistributionFeature(BaseEstimator):
	"""
	Gives a score that characterizes the distribution of a user's stars in the 5 possible
	bins.
	"""

	def __init__(self, data=None):
		self.data = data

	def fit(self, X, y):
		user_review_stars = defaultdict(Counter)
		for review in self.data.all_reviews().values():
			user_review_stars[review['user_id']].update([review['stars']])

		self.user_stars_distribution = {}
		for user_id, stars_counter in user_review_stars.iteritems():
			total_count = sum(stars_counter.values())
			# reviews can be grouped in exactly 5 bins by their stars
			bins_count = 5
			average_bin_size = float(total_count) / bins_count
			bins_deviation = [abs(stars_counter[i] - average_bin_size) 
								for i in range(1, bins_count + 1)]
			score = sum(bins_deviation) / total_count
			self.user_stars_distribution[user_id] = score
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			feature = self.user_stars_distribution[review['user_id']]
			feature_matrix.append([feature])
		return feature_matrix
