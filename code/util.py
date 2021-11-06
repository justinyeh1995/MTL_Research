import re
import pickle
import sys
import time
from nltk.corpus import stopwords
from scipy.stats import ttest_ind
import nltk 
nltk.download('stopwords')
nltk.download('punkt')

stop = set(stopwords.words('english'))

def normalizeString(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r" : ", ":", string)
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " ( ", string) 
	string = re.sub(r"\)", " ) ", string) 
	string = re.sub(r"\?", " ? ", string) 
	string = re.sub(r"\s{2,}", " ", string)   
	return string.strip().lower()

def removeStop(word):
	if word not in stop:
		return word

def key2value( target, dic ):
	return dic[target]

def write2file( file_name, data ):
	with open(  file_name, 'w') as f:
		for i in range(len(data)):
			f.write('{}\n'.format(data[i]))

def ttest( m1, m2 ):
	stat, p = ttest_ind(m1, m2)
	return stat, p

def write2pickle( data, fname='fname.pkl'):
	f_ = open( fname , 'wb')
	pickle.dump(data, f_)
	f_.close()

def loadPickle( fname ):
	with open(fname, 'rb') as f:
		return pickle.load(f)

def padding( x, max_len=256, padding_item=[0] ):
	return x+(max_len-len(x))*padding_item if len(x) < max_len else x[:max_len] 

def train_test_split( data, test_size=0.2 ):
	x, y = data
	test_len = int(test_size*len(x))

	train_x = x[test_len:]
	train_y = y[test_len:]
	test_x = x[:test_len]
	test_y = y[:test_len]

	return (train_x, train_y), (test_x, test_y)
