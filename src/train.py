from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import csv
from ml_metrics import rmsle
import pylab
import numpy

from data import Data
from estimators import *


def score(actual, prediction):
	return rmsle(actual, prediction.clip(0))


def plotPrediction(X, y, prediction):
	print "Plotting"
	X = numpy.array(X)[:,0]
	pylab.scatter(X, y, color='black')
	#Values for the X axis need to be sorted for a meaningful prediction line
	x_list, y_list = zip(*sorted(zip(X, prediction)))
	pylab.plot(x_list, y_list, color='blue', linewidth=3)
	pylab.show()



if __name__ == "__main__":
	data = Data()
	reviews = [review for key, review in data.training_reviews.iteritems()]
	review_votes = [review['votes']['useful'] for key, review in data.training_reviews.iteritems()]

	print "splitting training set"
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(reviews, review_votes, 
													train_size=0.6, test_size=0.4, random_state=0)


	# There is a bug in joblib that prevents us from spawning multiple jobs.
	# Paralelizing these features seems to work slower for now anyway.
	feature_union = FeatureUnion([
									('review_age', ReviewAgeEstimator()),
									('paragraphs_count', ParagraphCountEstimator()),
									('user_review_count', UserReviewCountEstimator(data)),
									('rev_length', ReviewLengthEstimator()),
									])

	pipeline = Pipeline([('features', feature_union),
						('scale', StandardScaler()),
						('sgdr', SGDRegressor())])
	pipeline.set_params(sgdr__n_iter=1000)

	print "fitting"
	pipeline.fit(X_train, y_train)
	print "predicting"
	prediction = pipeline.predict(X_test)
	# A review can't have a negative number of votes
	prediction = prediction.clip(0)
	print score(y_test, prediction)


	# sample plotting
	# estimator = ReviewLengthEstimator()
	# plotPrediction(estimator.transform(X_test), y_test, prediction)


	# Predict on Yelp's test set
	print "predicting test set"
	reviews = [review for key, review in data.test_reviews.iteritems()]
	prediction = pipeline.predict(reviews)
	prediction = prediction.clip(0)

	with open('../derivedData/submission.csv','wb') as csvfile:
		writer = csv.writer(csvfile)
		for (i, p) in enumerate(prediction):
			review_id = reviews[i]['review_id']
			writer.writerow([review_id, p])
