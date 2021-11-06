import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import MultiHeadAtt as attention

## Huggin face - Tranformers ##
from transformers import BertModel, BertConfig
import args
arg = args.process_command()
lang = arg.lang
gpu = arg.gpu

if lang == 'en':
	weight = 'bert-base-cased'
else:
	weight = 'cl-tohoku/bert-base-japanese-whole-word-masking'


#torch.cuda.set_device(1)
embed_size = 300
gamma = 1

class BERT(nn.Module):
	
	def __init__(self, bert_model = BertModel, bert_weight = weight, classes=2):
		super(BERT, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.MSELoss()
		
		# Main
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

		self.dropout = nn.Dropout(p=0.1)
		self.final = nn.Linear(768, 1)
		#self.final01 = nn.Linear(10, 1)	

		# Aux
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

		self.dropout_aux = nn.Dropout(p=0.1)
		self.tolower = nn.Linear(1536, 768)
		self.final_aux = nn.Linear(768, 1)
		#self.final_aux01 = nn.Linear(10, 1)	

	def loss_func(self):
		return self.loss1

	'''def main_task(self, data, doc1, doc2, count, cls_dict, mode='train'):'''
	#def main_task(self, data, doc1, doc2, count, score, mode='train'):
	def main_task(self, data, doc1, doc2, count, mode='train'):
		# content_ids
		cls_head_conts = []
		ratings = []
		tk = []
		## dictionary at batch level ## 
		cls_dict = {}
		count = count.squeeze().tolist()
		'''print(count)'''
		'''print(cls_dict)'''	
		not_duplicate = []
		## one batch ##
		for i, d in enumerate(data):
			#token_ids, attn_mask, seg_ids = d
			if len(list(d.size())) > 1:
				token_ids = d[0]
			else:
				token_ids = d
			token_ids = token_ids.view(1,302)
			hidden_reps, cls_head_cont = self.pretrained(token_ids)
			cls_head_conts.append(cls_head_cont)
			#if mode == 'train':
			#cls_head_cont = self.dropout(cls_head_cont)
			rating = self.final(cls_head_cont)
			ratings.append(rating)
			#print(cls_head_cont)
			'''need fix'''
			#if type(count) is not list or len(count) == len(set(count)):
			#	cls_head_cont = self.dropout(cls_head_cont)
			#	rating = self.final(cls_head_cont)
			#	ratings.append(rating)
			#	cls_head_conts.append(cls_head_cont)
			#		 
			#else:
			#	if cls_dict.get(count[i]) == None:	
			#		cls_dict[count[i]] = cls_head_cont
			#		#print(cls_dict[count[i]])
			#	cls_head_conts.append(cls_dict.get(count[i]))
			#	#print(cls_head_cont)
			#	#cls_head_cont = self.dropout(cls_dict[count[i]])
			#	rating = self.final(cls_head_cont)
			#	#print(rating)
			#	if count[i] not in not_duplicate:
			#		not_duplicate.append(count[i])
			#		ratings.append(rating)
			#	'''cls_head_conts.append(cls_head_cont)'''
			#	'''tk.append(token_ids)'''
		# torch.Size([batch_size])
		ratings = torch.cat(ratings, dim=1)
		#print(ratings)

		if mode == 'train':
			rating1 = []
			#print(len(doc1)) # 32: batch-size
			#for i, doc in enumerate(doc1):			
			for i, d in enumerate(data):
				# aux_ids
				#token_ids, attn_mask, seg_ids = doc
				
				aux_ids = d[1:]
				for aux in aux_ids:
					aux = aux.view(1,302)
					hidden_reps, cls_head_aux = self.pretrained_aux(aux)
					cls_head_cat = torch.cat([ cls_head_conts[i], cls_head_aux ], 1)
					cls_head_cat = self.tolower(cls_head_cat)
					#cls_head_cat = self.dropout_aux(cls_head_cat)
					rating = self.final_aux(cls_head_cat)
					rating1.append(rating)
				'''
				token_ids = token_ids.view(1,302)
				hidden_reps, cls_head_aux = self.pretrained_aux(token_ids)
				cls_head_cat = torch.cat([ cls_head_conts[i], cls_head_aux ], 1)
				cls_head_cat = self.tolower(cls_head_cat)
				cls_head_cat = self.dropout_aux(cls_head_cat)
				rating = self.final_aux(cls_head_cat)
				#if lang == 'en':
				#	rating = F.gelu(rating)
				#rating = self.final_aux01(rating)
				#if lang == 'en':			
				#	rating = F.gelu(rating)
				rating1.append(rating)
				'''
			rating1 = torch.cat(rating1, dim = 1)
			'''return ratings[0], rating1[0]#, cls_dict'''
			return ratings[0], rating1[0]

		else:
			'''return ratings[0], cls_dict'''
			return ratings[0]

	'''def forward(self, data_, cls_dict, mode='train'):'''
	def forward(self, data_, mode='train'):
		doc_org = data_[4]
		#y = data_[1]
		count = data_[6]
		score = data_[9]
		if mode == 'train':
			#aux1 = data_[5]
			#aux2 = None
			#label = data_[3]
			count = data_[6]	
			'''y_rating, y_label, cls_dict = self.main_task(doc_org, aux1, aux2, count, cls_dict, mode=mode)'''
			y_rating, y_label = self.main_task(doc_org, None, None, count, mode=mode)
			#y_rating, y_label = self.main_task(doc_org, aux1, aux2, count, score, mode=mode)
			#print(self.loss1(y_rating, y))
			#print(self.loss2(y_label.squeeze(), label))
			#print(y_label)
			#print(label)
			'''y_rating.cpu().numpy()'''
			#y.cpu().numpy()
			#y_new = []
			#not_duplicate = []
			#print(len(y))
			#print(len(count))
			#for i, c in enumerate(count):
			#	if c not in not_duplicate:
			#		not_duplicate.append(c)
			#		y_new.append(y[i].item())
			#'''print(y_new)'''	
			#y = torch.Tensor(y_new).cuda()
			#'''print(count)'''
			#'''print(y_rating)'''
			#'''print(y)'''
			
			ys = [s[0] for s in score]
			ys = torch.Tensor(ys).cuda(gpu)
			## rethink y label & label##
			labels = []
			for s in score:
				labels.extend(s[1:])
			labels = torch.tensor(labels).cuda(gpu)
			return self.loss1(y_rating, ys) + gamma*self.loss2(y_label.squeeze(), labels)# 
			
			#return  self.loss1(y_rating, y) #+ gamma*self.loss2(y_label.squeeze(), label)#, cls_dict
			#self.loss1(y_rating, y)# + gamma*self.loss2(y_label.squeeze(), label), cls_dict
		else:
			'''
			y_rating, cls_dict = self.main_task(doc_org, None, None, count, cls_dict, mode=mode)
			#print(y_rating)
			return y_rating.view(y_rating.size(0),) , cls_dict 
			'''
			y_rating = self.main_task(doc_org, None, None, count, mode=mode)
			#print(y_rating)
			return y_rating.view(y_rating.size(0),) 

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.MSELoss()

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
		
			return rating, rating1
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0]
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2]
			aux2 = None
			label = data_[3]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)

			return self.loss1(y_rating, y) + gamma*self.loss2(y_label.squeeze(), label)
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.MSELoss()

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
			
			return rating, rating1
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0].unsqueeze(1)
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2].unsqueeze(1)
			aux2 = None
			label = data_[3]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y) + gamma*self.loss2(y_label.squeeze(), label)	
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.MSELoss()

		self.rnn0 = nn.GRU(embed_size, 10, 1, bidirectional=True)
		
		self.doc1 = nn.Linear(10*2, 10*2)
		self.final = nn.Linear(10*2, 1)
		
		self.rnn1 = nn.GRU(embed_size, 10, 1, bidirectional=True)
		self.doc11 = nn.Linear(10*2*2, 10*2)
		self.final1 = nn.Linear(10*2, 1)
	
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
			
			return rating, rating1
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0]
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2]
			aux2 = None
			label = data_[3]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y) + gamma*self.loss2(y_label.squeeze(), label)	
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)

class Att(nn.Module):
	def __init__(self):
		super(Att, self).__init__()
		self.loss1 = nn.MSELoss()
		self.loss2 = nn.MSELoss()

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
			
			return rating, rating1
		else:
			return rating

	def loss_func(self):
		return self.loss1

	def forward(self, data_, mode='train'):
		doc_org = data_[0]
		y = data_[1]
		
		if mode == 'train':
			aux1 = data_[2]
			aux2 = None
			label = data_[3]
	
			y_rating, y_label = self.main_task(doc_org, aux1, aux2, mode=mode)
			
			return self.loss1(y_rating, y) + gamma*self.loss2(y_label.squeeze(), label)	
		else:
			y_rating = self.main_task(doc_org, None, None, mode=mode)
			
			return y_rating.view(y_rating.size(0),)
