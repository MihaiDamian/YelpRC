from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import csv
from ml_metrics import rmsle

from data import Data
from estimators import ReviewLengthEstimator
from estimators import UnigramEstimator
from estimators import UserReviewCountEstimator


def score(actual, prediction):
	return rmsle(actual, prediction)


if __name__ == "__main__":
	data = Data()
	reviews = [review for key, review in data.training_reviews.iteritems()]
	review_votes = [review['votes']['useful'] for key, review in data.training_reviews.iteritems()]

	print "splitting training set"
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(reviews, review_votes, 
													train_size=0.6, test_size=0.4, random_state=0)


	# there is a bug in joblib that prevents us from spawning multiple jobs 
	feature_union = FeatureUnion([#('unigram', UnigramEstimator()),
									('user_review_count', UserReviewCountEstimator(data)),
									('rev_length', ReviewLengthEstimator())])

	pipeline = Pipeline([('features', feature_union),
						('scale', StandardScaler()),
						('sgdr', SGDRegressor())])
	pipeline.set_params(sgdr__n_iter=1000, sgdr__eta0=0.00000001, 
						sgdr__learning_rate='constant', scale__with_mean=False)

	print "fitting"
	pipeline.fit(X_train, y_train)
	print "predicting"
	prediction = pipeline.predict(X_test)
	# A review can't have a negative number of votes
	prediction = prediction.clip(0)
	print score(y_test, prediction)


	# Predict on Yelp's test set
	"""print "predicting test set"
	reviews = [review for key, review in data.test_reviews.iteritems()]
	prediction = pipeline.predict(reviews)

	with open('../derivedData/submission.csv','wb') as csvfile:
		writer = csv.writer(csvfile)
		for (i, p) in enumerate(prediction):
			review_id = reviews[i]['review_id']
			writer.writerow([review_id, p])"""
