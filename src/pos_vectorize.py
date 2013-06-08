from nltk.tag import pos_tag
from nltk import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import cPickle

from data import Data


DATA_PATH = '../derivedData/posVectorizedReviews.pkl'


def loadData():
	"""
	Loads POS tag frequencies
	"""
	with open(DATA_PATH, "rb") as data_file:
		data = cPickle.load(data_file)
	return data


def vectorize():
	input_data = Data()
	training_reviews = [review for key, review in input_data.training_reviews.iteritems()]
	test_reviews = [review for key, review in input_data.test_reviews.iteritems()]
	all_reviews = training_reviews + test_reviews

	# Extract the tag counts
	# maps from tag to row index
	tag_indexes = {}
	# keeping the rows in a list because we don't no the maximum row lenght yet
	row_list = []
	for review in all_reviews:
		# allocate zeros for all known tags; new tags will just get appended
		row = [0] * len(tag_indexes)
		for sentence in sent_tokenize(review['text']):
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

	print "Saving data"
	training_rows_count = len(test_reviews)
	columns_count = len(tag_indexes)
	output_data = {'tag_indexes':tag_indexes, 'reviews':{}}
	# Make sure all rows have the same number of tags
	for index, row in enumerate(row_list):
		row = row + [0] * (columns_count - len(row))
		review_id = all_reviews[index]['review_id']
		output_data['reviews'][review_id] = [float(f) for f in row]
	with open(DATA_PATH, "wb") as data_file:
		cPickle.dump(output_data, data_file)
	

if __name__ == "__main__":
	vectorize()
