from fastapi import FastAPI
import threading
import requests

if __name__ == '__main__':
	from config import Config
else:
	from .config import Config

# Vector database API
class VDB:
	# define initialization function 
	def __init__(self, config, online: bool = False):
		self.config = config
		print(self.is_server_up())

		if self.is_server_up() == False and online != True:
			print('Offline')
			from sentence_transformers import SentenceTransformer, util
			from deep_translator import GoogleTranslator
			from faiss import IndexFlatL2
			import numpy as np
			import pickle

			self.model = SentenceTransformer(config.EMBEDDING_MODEL)
			self.vocab_file = config.VOCAB_FILE
			self.index_file = config.INDEX_FILE
			self.index = IndexFlatL2(self.model.encode('hello').shape[-1])
			self.translator = GoogleTranslator(source='auto', target='en')
			self.server = threading.Thread(target=self.intialize_server_listener)
			self.server.start()
			self.server_is_online = False
		else:
			print('Online')
			self.server_is_online = True

		self.vocab = []
		self.vocab_nontranslated = []
	
	def is_server_up(self):
		try:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + '/online', timeout=2)

			if resp.status_code == 200:
				return True
			else:
				return False
		except:
			return False
	
	def intialize_server_listener(self):
		app = FastAPI()

		@app.get("/online")
		async def online():
			return 200

		@app.get("/load")
		async def load(file_name: str = None):
			self.load(file_name)
			return 200

		@app.get("/save")
		async def save(file_name: str = None):
			self.save(file_name)
			return 200
			
		@app.get("/reset_index")
		async def reset_index():
			self.reset_index()
			return 200

		@app.get("/add")
		async def add(note: str):
			self.add(note)
			return 200

		@app.get("/remove")
		async def remove(idx: str):
			try:
				self.remove(int(idx))
			except:
				self.remove(idx)
			return 200

		@app.get("/confidence")
		async def confidence(note: str, exact: bool = True, confidence_threshold: float = 0.7):
			return self.confidence(note, exact, confidence_threshold)

		@app.get("/similar_str")
		async def similar_str(note: str):
			return self.similar_str(note)

		@app.get("/similar_idx")
		async def similar_idx(note: str):
			return self.similar_idx(note)

		@app.get("/fetch")
		async def fetch(idx: int):
			return self.fetch(idx)

		import uvicorn
		uvicorn.run(app, host='0.0.0.0', port=int(self.config.SERVER_PORT))
	
	# define API loading from given files
	def load(self, file_name: str = None):
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/load?file_name={file_name}', timeout=2)
		else:
			if file_name == None or file_name == 'None':
				try:
					with open(self.vocab_file, 'rb') as file:
						self.vocab = pickle.load(file)
					with open(self.vocab_file + 't', 'rb') as file:
						self.vocab_nontranslated = pickle.load(file)
					with open(self.index_file, 'rb') as file:
						self.index = pickle.load(file)
				except:
					pass
			else:
				try:
					with open(file_name.split('.')[0] + '.mmp', 'rb') as file:
						self.vocab = pickle.load(file)
					with open(file_name.split('.')[0] + '.mmpt', 'rb') as file:
						self.vocab_nontranslated = pickle.load(file)
					with open(file_name.split('.')[0] + '.idm', 'rb') as file:
						self.index = pickle.load(file)
				except:
					pass
			
	# define API saving to the given file
	def save(self, file_name: str = None):
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/save?file_name={file_name}', timeout=2)
		else:
			if file_name == None or file_name == 'None':
				with open(self.vocab_file, 'wb') as file:
					pickle.dump(self.vocab, file)
				with open(self.vocab_file + 't', 'wb') as file:
					pickle.dump(self.vocab_nontranslated, file)
				with open(self.index_file, 'wb') as file:
					pickle.dump(self.index, file)
			else:
				with open(file_name.split('.')[0] + '.mmp', 'wb') as file:
					pickle.dump(self.vocab, file)
				with open(file_name.split('.')[0] + '.mmpt', 'wb') as file:
					pickle.dump(self.vocab_nontranslated, file)
				with open(file_name.split('.')[0] + '.idm', 'wb') as file:
					pickle.dump(self.index, file)
	
	# define resetting the database index
	def reset_index(self):
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/reset_index', timeout=2)
		else:
			self.index.reset()

			for word in self.vocab:
				self.index.add(np.expand_dims(self.model.encode(word), 0))
	
	# define adding a new word-unit into the database
	def add(self, note: str):
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/add?note={note}', timeout=2)
		else:
			self.vocab.append(self.translator.translate(note))
			self.vocab_nontranslated.append(note)

			self.index.add(np.expand_dims(self.model.encode(note), 0))
	
	# define removing word-units by index \ words
	def remove(self, idx: int | str):
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/remove?idx={idx}', timeout=2)
		else:
			if type(idx) == int:
				note = self.vocab[idx]
				self.vocab = self.vocab[:idx] + self.vocab[idx+1:]
				self.vocab_nontranslated = self.vocab_nontranslated[:idx] + self.vocab_nontranslated[idx+1:]
			else:
				note = idx
				idx = self.vocab_nontranslated.index(idx)

				if idx != None:
					self.vocab = self.vocab[:idx] + self.vocab[idx+1:]
					self.vocab_nontranslated = self.vocab_nontranslated[:idx] + self.vocab_nontranslated[idx+1:]

			self.index.remove_ids(self.index.search(np.expand_dims(self.model.encode(note), 0), 1)[1][0])
	
	# fetch a similar word-unit (index) from the database
	def _sim(self, note: str) -> int:
		similar = self.index.search(np.expand_dims(self.model.encode(note), 0), 1)[1].squeeze()

		return int(similar)
	
	# return confidence level the given word-unit is present in the database
	def confidence(self, note: str, exact: bool = True, confidence_threshold: float = 0.7) -> float | bool:
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/confidence?note={note}&exact={exact}&confidence_threshold={confidence_threshold}', timeout=2)
			try:
				return float(resp.text)
			except:
				return bool(resp.text)
		else:
			idx = self._sim(note)
			note = self.translator.translate(note)

			if exact:
				return float(util.cos_sim(self.model.encode(note), self.model.encode(self.vocab[idx])).squeeze())
			else:
				return float(util.cos_sim(self.model.encode(note), self.model.encode(self.vocab[idx])).squeeze()) > confidence_threshold
	
	# return a similar word-unit
	def similar_str(self, note: str) -> str:
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/similar_str?note={note}', timeout=2)
			return resp.text[1:-1]
		else:
			idx = self._sim(note)
			return self.vocab_nontranslated[idx]
	
	# return an index of a similar word-unit
	def similar_idx(self, note: str) -> int:
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/similar_idx?note={note}', timeout=2)
			return int(resp.text)
		else:
			idx = self._sim(note)
			return idx
	
	# fetch a word-unit from the database by index
	def fetch(self, idx: int) -> str:
		if self.server_is_online == True:
			resp = requests.get(self.config.SERVER_URL + self.config.SERVER_PORT + f'/fetch?idx={idx}', timeout=2)
			return resp.text[1:-1]
		else:
			return self.vocab_nontranslated[idx]
		

if __name__ == '__main__':
	# intializes a vector database instance
	api = VDB(Config)


	# --- WRITE / READ OPERATIONS OVER THE DATABASE --- 


	# adds new word-units
	api.add('Hello world!')
	api.add('Hallo, wereld!')
	api.add('Привет мир!')

	# you can read the current vocabulary of word-units
	# print(api.vocab)

	# removes a word-unit by its index ('Hallo, wereld!' in this example)
	api.remove(1)

	# removes a word-unit by its string representation
	api.remove('Привет мир!')


	# --- SIMILARITY SEARCH AND CONFIDENCE LEVELS --- 


	# prints confidence level (float) of word-unit being present in the database
	print(api.confidence('Привет мир!'))

	# prints confidence (True or False) of word-unit being present in the database if the confidence level exceeds the given confidence threshold 
	print(api.confidence('Привет мир!', exact=False, confidence_threshold=0.5))
	print(api.confidence('Привет мир!', exact=True, confidence_threshold=0.5))

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
