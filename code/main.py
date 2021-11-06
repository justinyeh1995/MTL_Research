import numpy as np
import args
import data_handler as dh
import load_word2vec as wv
import math
import os 
import random
import sys
import torch
import torch.utils.data as Data
import training_handler
import csv

import MT_model
import MT_model_reg
import ST_model

from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

def load_data( data ):
	x, y, ids, id3, id4, id5 = dh.load_data( data )
	x_data_type = torch.FloatTensor 
	y_data_type = torch.FloatTensor 
	ids_data_type = torch.LongTensor
		
	data = Variable(x_data_type(x))
	target = Variable(y_data_type(y))
	input_ids = Variable(ids_data_type(ids))
	id3 = Variable(ids_data_type(id3))
	id4 = Variable(ids_data_type(id4))
	id5 = Variable(ids_data_type(id5))

	torch_dataset = Data.TensorDataset(data, target, input_ids, id3, id4, id5)
	return torch_dataset

def load_reg_data( data, sample_size ):
	x1, x2, y1, y2, cont_ids, aux_ids = dh.regression_data(data, sample_size)
		
	x1_data_type =  torch.LongTensor if type(x1[0]) == int else torch.FloatTensor 
	y1_data_type =  torch.LongTensor if type(y1[0]) == int else torch.FloatTensor 
	x2_data_type =  torch.LongTensor if type(x2[0]) == int else torch.FloatTensor 
	y2_data_type =  torch.LongTensor if type(y2[0]) == int else torch.FloatTensor 
	cont_data_type = torch.LongTensor
	aux_data_type = torch.LongTensor
	
	x1 = Variable(x1_data_type(x1))
	y1 = Variable(y1_data_type(y1))
	x2 = Variable(x2_data_type(x2))
	y2 = Variable(y2_data_type(y2))
	cont_ids = Variable(cont_data_type(cont_ids))
	aux_ids = Variable(aux_data_type(aux_ids))

	torch_dataset = Data.TensorDataset(x1, y1, x2, y2, cont_ids, aux_ids)
	
	return torch_dataset

def load_RANK_data( data, sample_size ):
	x1, x2, x3, y1, y2, cont_ids, aux1_ids, aux2_ids = dh.rankNet_data(data, sample_size)

	x1_data_type =  torch.LongTensor if type(x1[0]) == int else torch.FloatTensor 
	y1_data_type =  torch.LongTensor if type(y1[0]) == int else torch.FloatTensor 
	x2_data_type =  torch.LongTensor if type(x2[0]) == int else torch.FloatTensor 
	x3_data_type =  torch.LongTensor if type(x3[0]) == int else torch.FloatTensor 
	y2_data_type =  torch.LongTensor if type(y2[0]) == int else torch.FloatTensor 
	cont_data_type = torch.LongTensor
	aux_data_type = torch.LongTensor
	
	x1 = Variable(x1_data_type(x1))
	y1 = Variable(y1_data_type(y1))
	x2 = Variable(x2_data_type(x2))
	x3 = Variable(x2_data_type(x3))
	y2 = Variable(y2_data_type(y2))
	cont_ids = Variable(cont_data_type(cont_ids))
	aux1_ids = Variable(aux_data_type(aux1_ids))
	aux2_ids = Variable(aux_data_type(aux2_ids))

	torch_dataset = Data.TensorDataset(x1, y1, x2, x3, y2, cont_ids, aux1_ids, aux2_ids)
	
	return torch_dataset

def torch_data( torch_dataset, batch_size=64):
	N = len(torch_dataset)	
	print( 'total data size:\t{}'.format(N) )

	indices = list(range(N))
	split = int(math.floor(0.1*N))
	
	train_idx, valid_idx, test_idx = indices[split*2:], indices[split:split*2], indices[:split]

	train_sampler = SequentialSampler(train_idx)
	valid_sampler = SequentialSampler(valid_idx)
	test_sampler = SequentialSampler(test_idx)

	train_loader = Data.DataLoader( dataset=torch_dataset, batch_size=batch_size, sampler=train_sampler )
	valid_loader = Data.DataLoader( dataset=torch_dataset, batch_size=batch_size, sampler=valid_sampler)
	test_loader = Data.DataLoader( dataset=torch_dataset, batch_size=batch_size, sampler=test_sampler)

	return train_loader, valid_loader, test_loader

def RUN( model_, data_, model_name ):
	train_loader, valid_loader, test_loader = data_

	total = sum(p.numel() for p in model_.parameters() if p.requires_grad)
	print('# of para: {}'.format(total))	
	
	thandler.train( model_, train_loader, valid_loader, model_name )
	y_true, y_pred, avg_loss = thandler.test( model_, test_loader, model_name )

	return (avg_loss ** 0.5)

def main_process( data, models, encoder, path='./'):
	try:
		if os.path.exists(path) == False:
			os.mkdir(path)
	
		rmse_mlp = RUN( models.MLP(), data, path+'MLP.pt' )
		rmse_cnn = RUN( models.CNN(), data, path+'CNN.pt' )
		rmse_rnn = RUN( models.RNN(), data, path+'RNN.pt' )
		rmse_att = RUN( models.Att(), data, path+'Att.pt' )
		rmse_bert = RUN(models.BERT(), data, path+'BERT.pt')
		return rmse_mlp, rmse_cnn, rmse_rnn, rmse_att, rmse_bert

	except OSError:
		print('Create directory failed')
		return 0, 0, 0, 0

def print_result( lst1, lst2 ):
	print('==========')
	print('{}:\t\t{}\t{}'.format('MLP', lst1[0], lst2[0] ))
	print('{}:\t\t{}\t{}'.format('CNN', lst1[1], lst2[1] ))
	print('{}:\t\t{}\t{}'.format('RNN', lst1[2], lst2[2] ))
	print('{}:\t{}\t{}'.format('self-att', lst1[3], lst2[3] ))
	print('{}:\t\t{}\t{}'.format('BERT', lst1[4], lst2[4] ))

if __name__ == '__main__':
	thandler = training_handler.handler(args.process_command())	
	batch_size = thandler.batch_size
	sample_size = thandler.sample_size
	lang = thandler.lang
    encoder = thandler.encoder
	training = thandler.training
	task = thandler.task 

	if lang == 'en':
		dh.set_embedding(wv.word_embedding_en('../pretrained_embedding/model.bin'))
	else:
		dh.set_embedding(wv.word_embedding_jp('../pretrained_embedding/jawiki_20180420_300d.pkl'))

	jsons = os.listdir(training)
	data = (['{}/{}'.format(training, i) for i in jsons])
	data_ST = torch_data(load_data(data), batch_size=batch_size)
	if task == 'reg':
		data_MT = torch_data(load_reg_data(data, sample_size), batch_size=batch_size)
	else:
		data_MT = torch_data(load_RANK_data(data, sample_size), batch_size=batch_size)

	print('===> Single task')
	ST_result = main_process( data_ST, ST_model, '__ST_model__/', encoder )

	print('===> Multi task')
	if task == 'reg':
		MT_result = main_process( (data_MT[0], data_MT[1], data_ST[2]), MT_model_reg, '__MT_model__/', encoder )
	else:
		MT_result = main_process( (data_MT[0], data_MT[1], data_ST[2]), MT_model, '__MT_model__/' , encoder)

	print_result( list(ST_result), list(MT_result) )
