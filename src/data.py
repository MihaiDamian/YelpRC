import json
from os import path


class Data(object):
	"""Loads data from Yelp dumps"""

	#TODO: all datasets except reviews should be merged in a single dataset
	#this is because of missing data in the training and test sets

	def __init__(self, dataset="training_set"):
		"""
		dataset - "training_set" or "test_set"
		"""
		self.reviews = {}
		self.businesses = {}
		self.users = {}
		self.dataset = dataset

		self.__loadData()


	def __loadDataFromFile(self, file_name, key_name):
		"""
		file_name - file to load from
		key_name - field to index in dictionary by
		"""
		print "Reading " + file_name

		read_dict = {}
		dataset_folder = ""
		if self.dataset is "training_set":
			dataset_folder = "yelp_training_set"
		elif self.dataset is "test_set":
			dataset_folder = "yelp_test_set"
		file_path = path.join('../providedData', dataset_folder, file_name)
		data_file = open(file_path, 'r')

		for entry_string in data_file.readlines():
			entry_dict = json.loads(entry_string)
			entry_id = entry_dict[key_name]
			read_dict[entry_id] = entry_dict

		return read_dict


	def __loadData(self):
		print "Loading data dump files"
		self.reviews = self.__loadDataFromFile('yelp_set_review.json', 'review_id')
		self.businesses = self.__loadDataFromFile('yelp_set_business.json', 'business_id')
		self.users = self.__loadDataFromFile('yelp_set_user.json', 'user_id')
