# define a configuration class for the VDB
class Config:
	EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
	VOCAB_FILE = './storage/vocab.mmp'
	INDEX_FILE = './storage/vocab.idm'
	SERVER_PORT = '8345'
	SERVER_URL = 'http://0.0.0.0:'
