from sentence_transformers import SentenceTransformer, util
from config import Config
from annoy import AnnoyIndex
from faiss import IndexFlatL2
import numpy as np
import pickle


# Vector database API
class VDB:
	# define initialization function 
	def __init__(self, config):
		self.model = SentenceTransformer(config.EMBEDDING_MODEL)
		self.vocab_file = config.VOCAB_FILE
		self.index_file = config.INDEX_FILE
		self.index = IndexFlatL2(self.model.encode('hello').shape[-1])
		self.vocab = []
	
	# define API loading from given files
	def load(self, file_name: str = None):
		if file_name == None:
			try:
				with open(self.vocab_file, 'rb') as file:
					self.vocab = pickle.load(file)
				with open(self.index_file, 'rb') as file:
					self.index = pickle.load(file)
			except:
				pass
		else:
			try:
				with open(file_name.split('.')[0] + '.mmp', 'rb') as file:
					self.vocab = pickle.load(file)
				with open(file_name.split('.')[0] + '.idm', 'rb') as file:
					self.index = pickle.load(file)
			except:
				pass
			
	# define API saving to the given file
	def save(self, file_name: str = None):
		if file_name == None:
			with open(self.vocab_file, 'wb') as file:
				pickle.dump(self.vocab, file)
			with open(self.index_file, 'wb') as file:
				pickle.dump(self.index, file)
		else:
			with open(file_name.split('.')[0] + '.mmp', 'wb') as file:
				pickle.dump(self.vocab, file)
			with open(file_name.split('.')[0] + '.idm', 'wb') as file:
				pickle.dump(self.index, file)
	
	# define resetting the database index
	def reset_index(self):
		self.index.reset()

		for word in self.vocab:
			self.index.add(np.expand_dims(self.model.encode(word), 0))
	
	# define adding a new word-unit into the database
	def add(self, note: str):
		self.vocab.append(note)
		self.index.add(np.expand_dims(self.model.encode(note), 0))
	
	# define removing word-units by index \ words
	def remove(self, idx: int | str):
		if type(idx) == int:
			note = self.vocab[idx]
			self.vocab = self.vocab[:idx] + self.vocab[idx+1:]
		else:
			note = idx
			idx = self.vocab.index(idx)

			if idx != None:
				self.vocab = self.vocab[:idx] + self.vocab[idx+1:]

		self.index.remove_ids(self.index.search(np.expand_dims(self.model.encode(note), 0), 1)[1][0])
	
	# fetch a similar word-unit (index) from the database
	def _sim(self, note: str) -> int:
		similar = self.index.search(np.expand_dims(self.model.encode(note), 0), 1)[1].squeeze()

		return int(similar)
	
	# return confidence level the given word-unit is present in the database
	def confidence(self, note: str, exact: bool = True, confidence_threshold: float = 0.7) -> float | bool:
		idx = self._sim(note)

		if exact:
			return float(util.cos_sim(self.model.encode(note), self.model.encode(self.vocab[idx])).squeeze())
		else:
			return float(util.cos_sim(self.model.encode(note), self.model.encode(self.vocab[idx])).squeeze()) > confidence_threshold
	
	# return a similar word-unit
	def similar_str(self, note: str) -> str:
		idx = self._sim(note)
		return self.vocab[idx]
	
	# return an index of a similar word-unit
	def similar_idx(self, note: str) -> int:
		idx = self._sim(note)
		return idx
	
	# fetch a word-unit from the database by index
	def fetch(self, idx: int) -> str:
		return self.vocab[idx]
		

if __name__ == '__main__':
	# intializes a vector database instance
	api = VDB(Config)


	# --- WRITE / READ OPERATIONS OVER THE DATABASE --- 


	# adds new word-units
	api.add('Hello world!')
	api.add('Hallo, wereld!')
	api.add('Привет мир!')

	# you can read the current vocabulary of word-units
	print(api.vocab)

	# removes a word-unit by its index ('Hallo, wereld!' in this example)
	api.remove(1)

	# removes a word-unit by its string representation
	api.remove('Привет мир!')


	# --- SIMILARITY SEARCH AND CONFIDENCE LEVELS --- 


	# prints confidence level (float) of word-unit being present in the database
	print(api.confidence('Привет мир!'))

	# prints confidence (True or False) of word-unit being present in the database if the confidence level exceeds the given confidence threshold 
	print(api.confidence('Привет мир!', exact=False, confidence_threshold=0.5))

	# prints the most similar word-unit present in the database
	print(api.similar_str('Привет мир!'))

	# prints the index of the most similar word-unit present in the database
	print(api.similar_idx('Привет мир!'))

	# resets the index (ensures FAISS indexer uses consistent indecies with the current vocabulary of word-units)
	api.reset_index()


	# --- SERIALIZATION --- 


	# saves the database into ('db.mmp' - vocabulary, 'db.idm' - indexer (will be created automatically))
	api.save('db.mmp')

	# if the file path is not specified, then it will write into the default file, specified in the config
	# the same applies to API loading
	api.save()

	# loads the database from ('db.mmp' - vocabulary, 'db.idm' - indexer (will be specified automatically))
	api.load('db.mmp')
