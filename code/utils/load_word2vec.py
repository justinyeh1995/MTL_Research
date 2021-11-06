import numpy as np
import nltk
import util
from wikipedia2vec import Wikipedia2Vec as jap_vec
from gensim.models import KeyedVectors
	
class word_embedding_en:
	def __init__(self, path='./pretrained_embedding/model.bin'):
		self.wv_model = KeyedVectors.load_word2vec_format(path, binary=True)

	def word2vec( self, w ):
		try:
			return self.wv_model[w]
		except KeyError:
			return np.array(300*[0.0])

	def sentence2vec( self, org ):
		doc = util.normalizeString(org)
		if len(doc) > 0:
			doc = nltk.word_tokenize(doc)
		
			output = list(map(self.word2vec, doc))
			output = util.padding(output, padding_item=[300*[0]])
		else:	
			output = np.array(256*[300*[0.0]])
			
		return output

class word_embedding_jp:
	def __init__(self, path='./pretrained_embedding/jawiki_20180420_300d.pkl'):
		self.wv_model = jap_vec.load(path)

	def word2vec( self, w ):
		try:
			return self.wv_model.get_word_vector(w)
		except KeyError:
			return np.array(300*[0.0])

	def sentence2vec( self, org ):
		doc = org.split(' ')
		if len(doc) > 0:
			output = list(map(self.word2vec, doc))
			output = util.padding(output, padding_item=[300*[0]])
		else:	
			output = np.array(256*[300*[0.0]])
			
		return output


