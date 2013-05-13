from data import Data
import numpy
import pylab
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn import cross_validation
from ml_metrics import rmsle
import csv


class FeatureScaler(object):

	def __init__(self):
		self.scaler = None


	def scaleFeatures(self, X):
		print "Preprocessing X matrix"
		if self.scaler is None:
			self.scaler = preprocessing.StandardScaler().fit(X)
		return self.scaler.transform(X)



def extractXMatrix(data):
	X_matrix = []
	for key, review in data.reviews.iteritems():
		text_length = float(len(review['text']))
		X_matrix.append([text_length, text_length**2])
		#business = data.businesses[review['business_id']]
		#X_matrix.append([float(business['review_count'])])
	return numpy.array(X_matrix)


def extractYVector(data):
	y_vector = []
	for key, review in data.reviews.iteritems():
		y_vector.append(review['votes']['useful'])
	return numpy.array(y_vector)


def extractReviewIDs(data):
	review_ids = []
	for key, review in data.reviews.iteritems():
		review_ids.append(review['review_id'])
	return review_ids


def preprocess(X):
	print "Preprocessing X matrix"
	if feature_scaler is None:
		feature_scaler = preprocessing.StandardScaler().fit(X)
	else:
		print "using featured scaler"
	return feature_scaler.transform(X)


def trainRegressionModel(X, y):
	print "Training regression model"
	model = SGDRegressor(n_iter=400)
	model.fit(X, y)
	return model


def plotPrediction(X, y, prediction):
	print "Plotting"
	pylab.scatter(X[:,0], y, color='black')
	#Values for the X axis need to be sorted for a meaningful prediction line
	x_list, y_list = zip(*sorted(zip(X[:,0], prediction)))
	pylab.plot(x_list, y_list, color='blue', linewidth=3)
	pylab.show()


def score(actual, prediction):
	return rmsle(actual, prediction)


def train(feature_scaler):
	data = Data("training_set")
	X = extractXMatrix(data)
	y = extractYVector(data)
	X = feature_scaler.scaleFeatures(X)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
	model = trainRegressionModel(X, y)
	prediction = model.predict(X_test)
	#plotPrediction(X_test, y_test, prediction)
	print score(y_test, prediction)
	return model


def predict_test_set(model, feature_scaler):
	data = Data("test_set")
	review_ids = extractReviewIDs(data)
	X = extractXMatrix(data)
	X = feature_scaler.scaleFeatures(X)
	prediction = model.predict(X)
	with open('../derivedData/submission.csv','wb') as csvfile:
		writer = csv.writer(csvfile)
		for (p, r) in zip(prediction, review_ids):
			writer.writerow([r, p])



if __name__ == "__main__":
	feature_scaler = FeatureScaler()
	model = train(feature_scaler)
	predict_test_set(model, feature_scaler)
