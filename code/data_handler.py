import json
import numpy as np
import random
from itertools import combinations, permutations
import torch
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import args
arg = args.process_command()
lang = arg.lang 
if lang == 'en':
	weight = 'bert-base-cased'
else:
	weight = 'cl-tohoku/bert-base-japanese-whole-word-masking'

tokenizer = BertTokenizer.from_pretrained(weight)
wv_model = None

def tokenize(raw_content):
	# raw content is a single description here
	
	MAX_LEN = 302
	
	tokens = tokenizer.tokenize(raw_content)
	if len(tokens) > 300:
		tokens = tokens[0:300]

	tokens = ['[CLS]'] + tokens + ['[SEP]']
	padded_tokens = tokens +['[PAD]' for _ in range(MAX_LEN-len(tokens))]

	attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens ]
	seg_ids = [0 for _ in range(len(padded_tokens))]
	sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
	#return [sent_ids, attn_mask, seg_ids]
	return sent_ids
	
def set_embedding(embedding):
	global wv_model
	wv_model=embedding

def loadJSON( JSON ):
	_ = json.load(JSON)

	#category
	#review_count
	#avg_rating_stars#
	#rating_stats
	#entity
	#reviews

	auxs = map(dict, _['spoiler_reviews'])
	aux, aux_rating = zip(*[(i['content'], i['rating']) for i in auxs])

	content = wv_model.sentence2vec(_['description'])
	raw_content = _['description']
	y = _['rating']

	return content, y, aux, aux_rating

def loadJSON_raw( JSON ):
	_ = json.load(JSON)
	raw_content = _['description']

	return raw_content

def load_data( jsons ):
	'''
	contents, ys, auxs, aux_ratings = zip(*list(map( loadJSON, map(open, jsons) )))
	input_ids = []
	count = []
	counter = 0
	for i in jsons:
		raw_content = loadJSON_raw(open(i))
		input_id = tokenize(raw_content)
		input_ids.append(input_id)
		count.append(counter)
		counter += 1
	'''
	contents = []
	ys = []
	input_ids = []
	count = []
	counter = 0
	for i in jsons:
		content, y, aux, aux_rating = loadJSON(open(i))
		raw_content = loadJSON_raw(open(i))
		input_id = tokenize(raw_content)
		ys.append(y)
		contents.append(content)
		input_ids.append(input_id)
		count.append(counter)
		counter += 1
	
	return contents, ys, input_ids, input_ids, input_ids, input_ids, count

def regression_data( jsons, sample_size ):
	contents = []
	aux1 = []
	ys = []
	labels = []
	cont_ids = []
	aux1_ids = []
	count = []
	score = []
	counter = 0
	for i in jsons:
		content, y, aux, aux_rating = loadJSON(open(i))
			
		aux_pair_org = list(zip(aux, aux_rating))

		div = []
		for r in aux_rating:
			div.append(abs(y-r))
					
		aux_pair_org = [a[0] for a in sorted(zip(aux_pair_org, div), key = lambda t: t[1])] 	
			
		raw_content = loadJSON_raw(open(i))
		input_id = tokenize(raw_content)
		cont_id = []
		cont_id.append(input_id)	
		s = []
		s.append(y)
		cont = []
		cont.append(content)
		
		for j in aux_pair_org[:sample_size]:
			cont.append(wv_model.sentence2vec(j[0]))
			cont_id.append(tokenize(j[0])) 
			s.append(j[1])
		
		while len(cont) < sample_size+1:
			if (sample_size+1 - len(cont)) > len(aux_pair_org[:sample_size]):
				cont.extend(wv_model.sentence2vec(j[0]) for j in aux_pair_org[:sample_size])
				cont_id.extend(tokenize(j[0]) for j in aux_pair_org[:sample_size]) 
				s.extend(j[1] for j in aux_pair_org[:sample_size])
			else:
				cont.extend(wv_model.sentence2vec(aux_pair_org[j][0]) for j in range(sample_size+1-len(cont)))
				cont_id.extend(tokenize(aux_pair_org[j][0]) for j in range(sample_size+1-len(cont_id))) 
				s.extend(aux_pair_org[j][1] for j in range(sample_size+1-len(s)))
		cont_ids.append(cont_id)
		score.append(s)
		contents.append(cont)
	# cont_ids matches aux1_ids
	#return contents, aux1, ys, labels, cont_ids, aux1_ids, count, score
	return contents, cont_ids, score

def rankNet_data( jsons, sample_size ):
	contents = []
	aux1 = []
	aux2 = []
	ys = []
	labels = []
	cont_ids = []
	aux1_ids = []
	aux2_ids = []

	count = []
	counter = 0

	for i in jsons:
		content, y, aux, aux_rating= loadJSON(open(i))
		
		aux_pair_org = list(zip(aux, aux_rating))
		aux_pair_rnd = list(zip(aux, aux_rating))
		aux_pair_rnd.reverse()
		#np.random.shuffle(aux_pair_rnd)
	
		raw_content = loadJSON_raw(open(i))
		input_id = tokenize(raw_content)
#		contents.append(content)

		for j in zip(aux_pair_org[:sample_size], aux_pair_rnd[:sample_size]):
			aux1.append(wv_model.sentence2vec(j[0][0]))
			aux2.append(wv_model.sentence2vec(j[1][0]))
			label = 0.0 if j[0][1]-j[1][1] < 0 else 1.0
			labels.append(label)
			ys.append(y)
			contents.append(content)
			cont_ids.append(input_id)
			aux1_ids.append(tokenize(j[0][0]))
			aux2_ids.append(tokenize(j[1][0]))
			count.append(counter)
		counter += 1
		### need to update to lower version ###
		'''
		
		aux_pair_org = list(zip(aux[:ssize], aux_rating[:ssize]))
		aux_pair = list(combinations(aux_pair_org, 2))
		plength = len(aux_pair)

		for j in aux_pair:
			aux1.append(sentence2vec(j[0][0]))
			aux2.append(sentence2vec(j[1][0]))

			label = 0.0 if j[0][1]-j[1][1] < 0 else 1.0
			labels.append(label)
			ys.append(y)
			contents.append(content)
		'''

	return contents, aux1, aux2, ys, labels, cont_ids, aux1_ids, aux2_ids, count

def testing_data( data ):
	content = wv_model.sentence2vec(open(data).read())
		
	return [contents], [0]
