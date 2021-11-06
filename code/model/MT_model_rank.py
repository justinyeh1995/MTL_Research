import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import MultiHeadAtt as attention
## Huggin face - Tranformers ##
from transformers import BertModel, BertConfig
import args
arg = args.process_command()
lang = arg.lang

if lang == 'en':
	weight = 'bert-base-cased'
else:
	weight = 'cl-tohoku/bert-base-japanese-whole-word-masking'

embed_size = 300
gamma = 1

class BERT(nn.Module):
	
	def __init__(self, bert_model = BertModel, bert_weight = weight, classes=2):
		super(BERT, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.BCEWithLogitsLoss()

		#self.activation = nn.ReLU()
		
		self.pretrained = bert_model.from_pretrained(bert_weight)

		for param in self.pretrained.parameters():
			param.requires_grad = False
		for param in self.pretrained.encoder.layer[-1].parameters():
			param.requires_grad = True
		for param in self.pretrained.encoder.layer[-2].parameters():
			param.requires_grad = True
#		for param in self.pretrained.encoder.layer[-3].parameters():
#			param.requires_grad = True
#		for param in self.pretrained.encoder.layer[-4].parameters():
#			param.requires_grad = True

		#self.decoder = nn.Linear(768, classes)
		self.dropout = nn.Dropout(p=0.1)		
		self.final = nn.Linear(768, 1)
		#self.final01 = nn.Linear(10, 1)

		self.pretrained_aux = bert_model.from_pretrained(bert_weight)

		for param in self.pretrained_aux.parameters():
			param.requires_grad = False
		for param in self.pretrained_aux.encoder.layer[-1].parameters():
			param.requires_grad = True
		for param in self.pretrained_aux.encoder.layer[-2].parameters():
			param.requires_grad = True
#		for param in self.pretrained_aux.encoder.layer[-3].parameters():
#			param.requires_grad = True
#		for param in self.pretrained_aux.encoder.layer[-4].parameters():
#			param.requires_grad = True

		#self.decoder = nn.Linear(768, classes)
		self.dropout_aux = nn.Dropout(p=0.1)
		self.tolower = nn.Linear(1536, 768)
		self.final_aux = nn.Linear(768, 1)
		#self.final_aux01 = nn.Linear(10, 1)
	
	def loss_func(self):
		return self.loss1

	'''def main_task(self, data, doc1, doc2, count, cls_dict, mode='train'):'''
	def main_task(self, data, doc1, doc2, count, mode='train'):
		# content_ids
		encoder = self.pretrained(data)
		hidden_reps, cls_head = encoder[0], encoder[1]
		ratings= self.final(cls_head)

		if mode == 'train':
			# aux_ids 
				
			encoder1 = self.pretrained_aux(doc1)
			hidden_reps, cls_head = encoder1[0], encoder1[1]
			rating1= self.final_aux(cls_head)

			encoder2 = self.pretrained_aux(doc2)
			hidden_reps, cls_head = encoder2[0], encoder2[1]
			rating2= self.final_aux(cls_head)

			rating_diff = rating1 - rating2
					
			'''return ratings[0], rating_diff[0], cls_dict'''
			return ratings[0], rating_diff[0]
		else:
			'''return ratings[0], cls_dict'''
			return ratings[0]


	'''def forward(self, data_, cls_dict, mode='train'):'''
	def forward(self, data_, mode='train'):
		doc_org = data_[5]
		y = data_[1]
		count = data_[8]

		if mode == 'train':
			aux1 = data_[6]
			aux2 = data_[7]
			label = data_[4]
			'''count = data_[8]'''

			'''y_rating, y_label, cls_dict = self.main_task(doc_org, aux1, aux2, count,cls_dict,  mode=mode)'''
			y_rating, y_label, cls_dict = self.main_task(doc_org, aux1, aux2, count, mode=mode)
			#print(self.loss1(y_rating, y))
			#print(self.loss2(y_label.squeeze(), label))
			#print(y_label)
			#print(label)
			return  self.loss1(y_rating, y) + gamma*self.loss2(y_label.squeeze(), label)
			'''return  self.loss1(y_rating, y)# + gamma*self.loss2(y_label.squeeze(), label), cls_dict'''
		else:
			'''y_rating, cls_dict = self.main_task(doc_org, None, None, count, cls_dict, mode=mode)'''
			y_rating, cls_dict = self.main_task(doc_org, None, None, count, mode=mode)
			#print(y_rating)
			'''return y_rating.view(y_rating.size(0),) , cls_dict''' 
			return y_rating.view(y_rating.size(0),)
			
class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.BCEWithLogitsLoss()

		self.doc0 = nn.Linear(300, 10)
		self.doc1 = nn.Linear(10, 10)
		self.final = nn.Linear(10, 1)
		
		self.doc01 = nn.Linear(300, 10)
		self.doc11 = nn.Linear(20, 10)
		self.final1 = nn.Linear(10, 1)
	
	def main_task(self, x, doc1, doc2, mode='train'):
		x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)
		x = self.doc0(x)
		x = F.relu(x)

		x1 = self.doc1(x)
		x1 = F.relu(x1)

		rating = self.final(x1)

		if mode == 'train':
			doc1 = F.avg_pool2d(doc1, (doc1.size(1), 1)).squeeze(1)
			doc1 = self.doc01(doc1)
			doc1 = F.relu(doc1)

			doc1 = self.doc11(torch.cat([x, doc1], 1))
			doc1 = F.relu(doc1)

			rating1 = self.final1(doc1)
			
			doc2 = F.avg_pool2d(doc2, (doc2.size(1), 1)).squeeze(1)
			doc2 = self.doc01(doc2)
			doc2 = F.relu(doc2)

			doc2 = self.doc11(torch.cat([x, doc2], 1))
			doc2 = F.relu(doc2)

			rating2 = self.final1(doc2)

			return rating, (rating1-rating2)
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0]
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2]
			aux2 = data_[3]
			label = data_[4]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y)+ gamma*self.loss2(y_label.squeeze(), label)
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.BCEWithLogitsLoss()

		self.conv0 = nn.Conv2d(1, 10, (3, 300), padding=(1, 0))
		
		self.doc1 = nn.Linear(10, 10)
		self.final = nn.Linear(10, 1)
		
		self.conv01 = nn.Conv2d(1, 10, (3, 300), padding=(1, 0))
		self.doc11 = nn.Linear(20, 10)
		self.final1 = nn.Linear(10, 1)
	
	def main_task(self, x, doc1, doc2, mode='train'):
		x = self.conv0(x)
		x = F.relu(x).squeeze(3)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		
		x1 = self.doc1(x)
		x1 = F.relu(x1)

		rating = self.final(x1)

		if mode == 'train':
			doc1 = self.conv01(doc1)
			doc1 = F.relu(doc1).squeeze(3)
			doc1 = F.max_pool1d(doc1, doc1.size(2)).squeeze(2)
		
			doc1 = self.doc11(torch.cat([x, doc1], 1))
			doc1 = F.relu(doc1)

			rating1 = self.final1(doc1)
			
			doc2 = self.conv01(doc2)
			doc2 = F.relu(doc2).squeeze(3)
			doc2 = F.max_pool1d(doc2, doc2.size(2)).squeeze(2)
		
			doc2 = self.doc11(torch.cat([x, doc2], 1))
			doc2 = F.relu(doc2)

			rating2 = self.final1(doc2)

			return rating, (rating1-rating2)
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0].unsqueeze(1)
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2].unsqueeze(1)
			aux2 = data_[3].unsqueeze(1)
			label = data_[4]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y)+ gamma*self.loss2(y_label.squeeze(), label)

		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.BCEWithLogitsLoss()

		self.rnn0 = nn.GRU(embed_size, embed_size, 1, bidirectional=True)
		
		self.doc1 = nn.Linear(embed_size*2, embed_size*2)
		self.final = nn.Linear(embed_size*2, 1)
		
		self.rnn1 = nn.GRU(embed_size, embed_size, 1, bidirectional=True)
		self.doc11 = nn.Linear(embed_size*2*2, embed_size*2)
		self.final1 = nn.Linear(embed_size*2, 1)
	
	def main_task(self, x, doc1, doc2, mode='train'):
		x, hidden = self.rnn0(x)
		x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)

		x1 = self.doc1(x)
		x1 = F.relu(x1)

		rating = self.final(x1)

		if mode == 'train':
			doc1, hidden = self.rnn1(doc1)
			doc1 = F.avg_pool2d(doc1, (doc1.size(1), 1)).squeeze(1)

			doc1 = self.doc11(torch.cat([x, doc1], 1))
			doc1 = F.relu(doc1)

			rating1 = self.final1(doc1)
			
			doc2, hidden = self.rnn1(doc2)
			doc2 = F.avg_pool2d(doc2, (doc2.size(1), 1)).squeeze(1)

			doc2 = self.doc11(torch.cat([x, doc2], 1))
			doc2 = F.relu(doc2)

			rating2 = self.final1(doc2)

			return rating, (rating1-rating2)
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0]
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2]
			aux2 = data_[3]
			label = data_[4]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y)+ gamma*self.loss2(y_label.squeeze(), label)	
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)

class Att(nn.Module):
	def __init__(self):
		super(Att, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.BCEWithLogitsLoss()

		self.att0 = attention(2, embed_size, 2)
		
		self.doc1 = nn.Linear(embed_size, embed_size)
		self.final = nn.Linear(embed_size, 1)
		
		self.att1 = attention(2, embed_size, 2)
		self.doc11 = nn.Linear(embed_size*2, embed_size)
		self.final1 = nn.Linear(embed_size, 1)

	def main_task(self, x, doc1, doc2, mode='train'):
		x = self.att0(x)
		x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)

		x1 = self.doc1(x)
		x1 = F.relu(x1)

		rating = self.final(x1)

		if mode == 'train':
			doc1 = self.att1(doc1)
			doc1 = F.avg_pool2d(doc1, (doc1.size(1), 1)).squeeze(1)

			doc1 = self.doc11(torch.cat([x, doc1], 1))
			doc1 = F.relu(doc1)

			rating1 = self.final1(doc1)
			
			doc2 = self.att1(doc2)
			doc2 = F.avg_pool2d(doc2, (doc2.size(1), 1)).squeeze(1)

			doc2 = self.doc11(torch.cat([x, doc2], 1))
			doc2 = F.relu(doc2)

			rating2 = self.final1(doc2)

			return rating, (rating1-rating2)
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0]
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2]
			aux2 = data_[3]
			label = data_[4]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y)+ gamma*self.loss2(y_label.squeeze(), label)	
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)


