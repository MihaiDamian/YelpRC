import ast


class Data(object):
	"""Loads data from Yelp dumps"""

	def __init__(self):
		# TODO: load the other dumps as well
		self.reviews = {}

		self.__loadData()


	def __loadData(self):
		print "Loading data dump files"

		reviewsFile = open('../providedData/yelp_training_set/yelp_training_set_review.json','r')

		for review_string in reviewsFile.readlines():
			review_dict = ast.literal_eval(review_string)
			review_id = review_dict['review_id']
			self.reviews[review_id] = review_dict