import json
from os import path


class Data(object):
	"""Loads data from Yelp dumps"""

	def __init__(self):
		# TODO: load the other dumps as well
		self.reviews = {}
		self.businesses = {}

		self.__loadData()


	def __loadDataFromFile(self, file_name, key_name):
		print "Reading " + file_name

		read_dict = {}
		file_path = path.join('../providedData/yelp_training_set', file_name)
		data_file = open(file_path, 'r')

		for entry_string in data_file.readlines():
			entry_dict = json.loads(entry_string)
			entry_id = entry_dict[key_name]
			read_dict[entry_id] = entry_dict

		return read_dict


	def __loadData(self):
		print "Loading data dump files"
		self.reviews = self.__loadDataFromFile('yelp_training_set_review.json', 'review_id')
		self.businesses = self.__loadDataFromFile('yelp_training_set_business.json', 'business_id')