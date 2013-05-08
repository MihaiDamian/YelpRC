from data import Data
import numpy
import pylab
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing


def buildXyStructures(data):
	print "Building X y matrices"

	X_matrix = []
	y_vector = []
	for key, review in data.reviews.iteritems():
		X_matrix.append([float(len(review['text']))])
		y_vector.append(review['votes']['useful'])
	return numpy.array(X_matrix), numpy.array(y_vector)


def preprocess(X):
	print "Preprocessing X matrix"
	return preprocessing.scale(X)


def trainRegressionModel(X, y):
	print "Training regression model"

	model = SGDRegressor(alpha=0.1, n_iter=20)
	model.fit(X, y)
	return model


def plotPrediction(X, y, model):
	print "Plotting"
	pylab.scatter(X[:,0], y, color='black')
	pylab.plot(X[:,0], model.predict(X), color='blue', linewidth=3)
	pylab.show()


data = Data()
X, y = buildXyStructures(data)
X = preprocess(X)
model = trainRegressionModel(X, y)
plotPrediction(X, y, model)