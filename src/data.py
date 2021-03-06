import json
from os import path
from datetime import datetime


DATE_FORMAT = '%Y-%m-%d'


class Data(object):
	"""Loads data from Yelp dumps"""

	def __init__(self):
		self.training_reviews = {}
		self.test_reviews = {}
		self.businesses = {}
		self.users = {}

		self.__loadData()


	def __load_to_dictionary(self, file_name, key_name, dataset_folder):
		"""
		file_name - file to load from
		key_name - field to index in dictionary by
		"""
		print "Reading " + file_name

		read_dict = {}
		file_path = path.join('../providedData', dataset_folder, file_name)
		data_file = open(file_path, 'r')

		for entry_string in data_file.readlines():
			entry_dict = json.loads(entry_string)
			entry_id = entry_dict[key_name]
			read_dict[entry_id] = entry_dict

		return read_dict


	def __decode_review_dates(self, review_dict):
		for review in review_dict.values():
			review['date'] = datetime.strptime(review['date'], DATE_FORMAT)


	def __loadData(self):
		print "Loading data dump files"
		self.training_reviews = self.__load_to_dictionary('yelp_set_review.json', 'review_id', 
														'yelp_training_set')
		self.test_reviews = self.__load_to_dictionary('yelp_set_review.json', 'review_id', 
														'yelp_test_set')
		self.__decode_review_dates(self.training_reviews)
		self.__decode_review_dates(self.test_reviews)

		# inject snapshot dates (these were communicated on the competition site)
		for review in self.training_reviews.values():
			review['snapshot_date'] = datetime.strptime("2013-01-19", DATE_FORMAT)
		for review in self.test_reviews.values():
			review['snapshot_date'] = datetime.strptime("2013-03-12", DATE_FORMAT)

		# Only review objects need to be kept separate; the others can be merged together to fill in 
		# missing associations

		# The test objects may contain less info than the training objects. For this reason the 
		# training set is loaded last to overwrite any existing objects from the test set

		self.businesses = self.__load_to_dictionary('yelp_set_business.json', 'business_id', 
													'yelp_test_set')
		self.businesses.update(self.__load_to_dictionary('yelp_set_business.json', 'business_id', 
														'yelp_training_set'))

		self.users = self.__load_to_dictionary('yelp_set_user.json', 'user_id', 'yelp_test_set')
		self.users.update(self.__load_to_dictionary('yelp_set_user.json', 'user_id', 
													'yelp_training_set'))

		self.training_checkins = self.__load_to_dictionary('yelp_set_checkin.json', 'business_id', 
													'yelp_training_set')
		self.test_checkins = self.__load_to_dictionary('yelp_set_checkin.json', 'business_id',
														 'yelp_test_set')


	def get_business_for_review(self, review):
		return self.businesses[review['business_id']]

	def all_reviews(self):
		all_reviews = {}
		all_reviews.update(self.training_reviews)
		all_reviews.update(self.test_reviews)
		return all_reviews
