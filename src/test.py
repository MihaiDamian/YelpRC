from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from data import Data
from train import score
	

class ReviewLengthTransformer(BaseEstimator):

	def fit(self, X, y):
		return self

	def transform(self, X):
		feature_matrix = []
		for review in X:
			text_length = float(len(review))
			feature_matrix.append([text_length, text_length**2])
		return feature_matrix


if __name__ == "__main__":
	data = Data("training_set")
	review_texts = [review['text'] for key, review in data.reviews.iteritems()]
	review_votes = [review['votes']['useful'] for key, review in data.reviews.iteritems()]

	print "splitting training set"
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(review_texts, review_votes, 
													train_size=0.6, test_size=0.4, random_state=0)



	unigram_pipe = Pipeline([('hash_vect', HashingVectorizer()), 
						('tfidf', TfidfTransformer())])

	#there is a bug in joblib that prevents us from spawning multiple jobs 
	feature_union = FeatureUnion(transformer_list=[('unigram_pipe', unigram_pipe), 
										('rev_length', ReviewLengthTransformer())])

	main_pipe = Pipeline([('features', feature_union),
						('scale', StandardScaler()),
						('sgdr', SGDRegressor())])
	main_pipe.set_params(sgdr__n_iter=1000, sgdr__eta0=0.00000001, 
						sgdr__learning_rate='constant', scale__with_mean=False)

	print "fitting"
	main_pipe.fit(X_train, y_train)
	print "predicting"
	prediction = main_pipe.predict(X_test)
	#A review can't have a negative number of votes
	prediction = prediction.clip(0)
	print score(y_test, prediction)
