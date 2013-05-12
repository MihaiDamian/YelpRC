from data import Data
import numpy
import pylab
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn import cross_validation
from ml_metrics import rmsle


def buildXyStructures(data):
	print "Building X y structures"

	X_matrix = []
	y_vector = []
	for key, review in data.reviews.iteritems():
		text_length = float(len(review['text']))
		X_matrix.append([text_length, text_length**2])
		#business = data.businesses[review['business_id']]
		#X_matrix.append([float(business['review_count'])])
		y_vector.append(review['votes']['useful'])
	return numpy.array(X_matrix), numpy.array(y_vector)


def preprocess(X):
	print "Preprocessing X matrix"
	return preprocessing.scale(X)


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


if __name__ == "__main__":
	data = Data()
	X, y = buildXyStructures(data)
	X = preprocess(X)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
	model = trainRegressionModel(X, y)
	prediction = model.predict(X_test)
	#plotPrediction(X_test, y_test, prediction)
	print score(y_test, prediction)
