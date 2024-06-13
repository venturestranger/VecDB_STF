# Vector Database (VecDB) API

This repository contains the implementation of a vector database API using the SentenceTransformer model for embeddings and FAISS for fast similarity search. The API allows for adding, removing, and querying word-units in a vector space, providing functionalities such as similarity search and confidence levels.

## Features

- Add new word-units to the vector database.
- Remove word-units by index or by string representation.
- Fetch word-units by index.
- Perform similarity search to find the most similar word-unit.
- Calculate confidence levels for the presence of word-units in the database.
- Save and load the database from files.
- Reset the index to ensure consistency with the current vocabulary.

## Installation

To use this API, ensure you have the following dependencies installed:

- `sentence-transformers`
- `faiss`
- `numpy`
- `pickle`

You can install these dependencies using pip:

```sh
pip install sentence-transformers faiss numpy
```

## Usage

### Initialization

First, initialize the vector database with a configuration object that specifies the embedding model, vocabulary file, and index file:

```python
from VecDB_STF import VDB
from VecDB_STF.config import Config

api = VDB(Config)
```

### Adding Word-Units

Add new word-units to the database:

```python
api.add('Hello world!')
api.add('Hallo, wereld!')
api.add('Привет мир!')
```

### Removing Word-Units

Remove word-units by index or by string representation:

```python
api.remove(1)  # Removes 'Hallo, wereld!' by index
api.remove('Привет мир!')  # Removes 'Привет мир!' by string
```

### Querying the Database

Fetch the current vocabulary of word-units:

```python
print(api.vocab)
```

Get the confidence level of a word-unit being present in the database:

```python
print(api.confidence('Привет мир!'))
print(api.confidence('Привет мир!', exact=False, confidence_threshold=0.5))
```

Find the most similar word-unit present in the database:

```python
print(api.similar_str('Привет мир!'))
```

Get the index of the most similar word-unit present in the database:

```python
print(api.similar_idx('Привет мир!'))
```

### Resetting the Index

Reset the index to ensure FAISS indexer uses consistent indices with the current vocabulary of word-units:

```python
api.reset_index()
```

### Saving and Loading the Database

Save the database to specified files:

```python
api.save('db.mmp')
```

If the file path is not specified, it will save to the default files specified in the config:

```python
api.save()
```

Load the database from specified files:

```python
api.load('db.mmp')
```

## Example

Here is a complete example of using the API:

```python
if __name__ == '__main__':
api = VDB(Config)

# Add word-units
api.add('Hello world!')
api.add('Hallo, wereld!')
api.add('Привет мир!')

# Print current vocabulary
print(api.vocab)

# Remove word-units
api.remove(1)
api.remove('Привет мир!')

# Print confidence levels
print(api.confidence('Привет мир!'))
print(api.confidence('Привет мир!', exact=False, confidence_threshold=0.5))

# Find similar word-units
print(api.similar_str('Привет мир!'))
print(api.similar_idx('Привет мир!'))

# Reset the index
api.reset_index()

# Save and load the database
api.save('db.mmp')
api.load('db.mmp')

# Save and load the database by a default path
api.save()
api.load()
```
