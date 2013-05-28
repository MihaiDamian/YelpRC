from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


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
		return super(UnigramEstimator, self).__init__([('review_text', ReviewTextTransformer()),
														('hash_vect', HashingVectorizer()), 
														('tfidf', TfidfTransformer())])


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
