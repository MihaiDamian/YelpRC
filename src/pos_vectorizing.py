from nltk.tag import pos_tag
from nltk import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import cPickle

from data import Data


TRAINING_MATRIX_PATH = '../derivedData/posVectorizedTrain.pkl'
TEST_MATRIX_PATH = '../derivedData/posVectorizedTest.pkl'


def saveMatrix(row_list, columns_count, filepath):
	matrix = []
	for row in row_list:
		row = row + [0] * (columns_count - len(row))
		matrix.append(row)
	with open(filepath, "wb") as matrix_file:
		cPickle.dump(matrix, matrix_file)


def loadMatrix(filepath):
	with open(filepath, "rb") as matrix_file:
		matrix = cPickle.load(matrix_file)
	for row in matrix:
		print row
	return matrix


def vectorize():
	data = Data()
	training_reviews = [review['text'] for key, review in data.training_reviews.iteritems()][:100]
	test_reviews = [review['text'] for key, review in data.test_reviews.iteritems()][:100]
	all_reviews = training_reviews + test_reviews

	# Extract the tag counts
	# maps from tag to row index
	tag_indexes = {}
	# keeping the rows in a list because we don't no the maximum row lenght yet
	row_list = []
	for review in all_reviews:
		# allocate zeros for all known tags; new tags will just get appended
		row = [0] * len(tag_indexes)
		for sentence in sent_tokenize(review):
			for token, tag in pos_tag(word_tokenize(sentence)):
				if tag in tag_indexes:
					index = tag_indexes[tag]
					row[index] += 1
				else:
					index = len(tag_indexes)
					tag_indexes[tag] = index
					row.append(1)
		row_list.append(row)
		print "Vectorizing review " + str(len(row_list))

	training_rows_count = len(test_reviews)
	columns_count = len(tag_indexes)
	print "Saving matrices"
	saveMatrix(row_list[:training_rows_count], columns_count, TRAINING_MATRIX_PATH)
	saveMatrix(row_list[training_rows_count:], columns_count, TEST_MATRIX_PATH)
	

if __name__ == "__main__":
	vectorize()
