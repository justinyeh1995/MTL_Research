import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import time
from earlyStopping import EarlyStopping

class handler:
	def __init__(self, arg):
		self.gpu = arg.gpu
		self.epoch_size = arg.epoch
		self.batch_size = arg.batch
		self.training = arg.train
		self.lang = arg.lang
		self.task = arg.task
		self.sample_size = arg.sample_size
		self.device = self.device_setting( self.gpu )
		self.hist = dict()
		self.print_para()
		self.file = arg.file

	def device_setting( self, gpu=1 ):
		return 'cpu' if gpu == -1 else 'cuda:{}'.format(gpu)

	def print_para( self ):
		print( 'epoch size:\t{}'.format(self.epoch_size) )	
		print( 'batch size:\t{}'.format(self.batch_size) )	
		print( 'sample size:\t{}'.format(self.sample_size) )	
		print( 'language:\t{}'.format(self.lang) )	
		print( 'task:\t\t{}'.format('Regression' if self.task == 'reg' else 'Ranking') )	
		print( 'training:\t{}'.format(self.training) )

	def print_hist( self ):
		string = '===> '

		for k in self.hist.keys():
			val = self.hist[k]

			if type(val) == float:
				string += '{}:{:.6f}\t'.format(k,val)	
			else:
				string += '{}:{}\t'.format(k,val)
		print(string)	
			
	def train( self, model_, train_loader, valid_loader, model_name='checkpoint.pt' ):
		es = EarlyStopping(patience=10, verbose=True)
		
		if model_.__class__.__name__ == 'BERT':
			print('lr=2e-5')
			optimizer = torch.optim.Adam(model_.parameters(), lr=2e-5)
			#if self.epoch_size > 30:
			#	self.epoch_size = 30
		else:
			print('lr=3e-4')
			optimizer = torch.optim.Adam(model_.parameters(), lr=3e-4)
		
		model_.to(self.device)
		loss_func = model_.loss_func()

		best_model = None
		print(f'Epoch Size: {self.epoch_size}')	
		for epoch in range(self.epoch_size):
			model_.train()
			start = time.time()
			train_loss, valid_loss = 0.0, 0.0
			train_acc = 0.0
			N_train = 0

			for i, data_ in enumerate(train_loader):
				data_ = [_.to(self.device) for _ in data_]
				optimizer.zero_grad()
				#y_pred = model_(data_, mode='')
				#loss = loss_func(y_pred, data_[1]) 
				loss = model_(data_)
				loss.backward()
				train_loss += loss.item()*len(data_[0])
				#train_acc += self.accuracy( y_pred, y_ ).item()*len(x_)
				optimizer.step()
				N_train += len(data_[0])
			
			self.hist['Epoch'] = epoch+1
			self.hist['time'] = time.time()-start
			self.hist['train_loss'] = train_loss/N_train	
			#self.hist['train_acc'] = train_acc/len(train_loader.dataset)

			torch.save(model_.state_dict(), model_name)
			
			if valid_loader != None:
				valid_true, valid_pred, valid_loss = self.test(model_, valid_loader, model_name)

				es(valid_loss, model_, model_name)
				
			self.print_hist()
			
			if es.early_stop:	
				print('Early stopping')
				break
				
	def test( self, model_, test_loader, model_name='checkpoint.pt' ):
		model_.load_state_dict(torch.load(model_name))
		model_.to(self.device)
		model_.eval()
		test_loss = 0.0
		test_acc = 0.0
		N_test = 0
		loss_func = model_.loss_func()

		y_pred, y_true = [], []
		with torch.no_grad():
			for i, data_ in enumerate(test_loader):
				data_ = [_.to(self.device) for _ in data_]
				logit = model_(data_, mode='test')
				loss = loss_func(logit, data_[1])
				y_pred.extend(logit)
				y_true.extend(data_[1])
				test_loss += loss.item()*len(data_[0])
				#test_acc += self.accuracy( logit, y_ ).item()*len(x_)
				N_test+=len(data_[0])
		self.hist['val_loss'] = test_loss/N_test
		#self.hist['val_acc'] = test_acc/len(test_loader.dataset)
		
		return y_true , y_pred, test_loss/N_test

	def predict( self, model_, dataset, model_name='checkpoint.pt' ):
		model_.to(self.device)
		model_.load_state_dict(torch.load(model_name))
		model_.eval()
		torch_dataset = self.load_data(dataset, mode='predict')
		print('# of predicted data : {}'.format(len(torch_dataset)))

		test_loader = Data.DataLoader( dataset=torch_dataset )
		_, y_pred, avg_loss = self.test( model_, test_loader )

		return y_pred
