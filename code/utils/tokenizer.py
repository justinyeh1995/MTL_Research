import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERT(nn.Module):
	
	def __init__(self, bert_model, bert_weight, classes=2):
		super(BERT, self).__init__()
		self.loss = nn.MSELoss()
		self.activation = nn.ReLU()
		
		self.pretrained = bert_model.from_pretrained(bert_weight)

		for param in self.pretrained.parameters():
			param.requires_grad = False
		for param in self.pretrained.encoder.layer[-1].parameters():
			param.requires_grad = True
		for param in self.pretrained.encoder.layer[-2].parameters():
			param.requires_grad = True
		for param in self.pretrained.encoder.layer[-3].parameters():
			param.requires_grad = True
		for param in self.pretrained.encoder.layer[-4].parameters():
			param.requires_grad = True

		self.decoder = nn.Linear(768, classes)
		self.dropout = nn.Dropout(p=0.1)
		self.final = nn.Linear(768, 1)
	
	def loss_func(self):
		return self.loss
	# Need to update the way data is handled
	def main_task(self, data):
		token_ids, attn_mask, seg_ids = data[0], data[1], data[2]
		hidden_reps, cls_head = self.pretrained(token_ids, attention_mask = attn_mask, token_type_ids = seg_ids)
		rating = self.final(cls_head)

		return cls_head, rating

	def forward(self, data , mode='train'):
		cls_head, y_rating = self.main_task(data)
		# Need to update data and loss
		if mode == 'train':
#			return self.loss(y_rating, [3.5])
			return y_rating
		else:
			return y_rating.view(y_rating.size(0),)

def tokenize(raw_content):
	# raw content is a single description here
	MAX_LEN = 128

	tokens = tokenizer.tokenize(raw_content)
	print(len(tokens[:100]))
	tokens = ['[CLS]'] + tokens + ['[SEP]']
	padded_tokens = tokens +['[PAD]' for _ in range(MAX_LEN-len(tokens))]

	attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
	seg_ids = [0 for _ in range(len(padded_tokens))]
	sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

	token_ids = torch.tensor(sent_ids).unsqueeze(0)
	attn_mask = torch.tensor(attn_mask).unsqueeze(0)
	seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
	
	return token_ids, attn_mask, seg_ids
			
token_ids, attn_mask, seg_ids = tokenize("In Landscape after the Battle, Andrzej Wajda in the second era of his filmmaking career, depicts emotional and psychologi    cal confusion in a former Nazi-prison in Poland, freed immediately after the WWII.<br /><br />A hand-held camera explores a lot of extreme close-ups and vivid colors. The end credit as graffiti on flanks of     freight train cars symbolically concludes the film. The soundtrack is great, except Vivaldi, which sounds tacky in pop-art fashion, in the opening sequence.")

data = [token_ids, attn_mask, seg_ids]
#print(data)
#bert = BertModel.from_pretrained("bert-base-cased")
bert = BERT(BertModel, "bert-base-uncased")
#sent_num = len(token_ids[0])
#print(sent_num)
#h = []
#cls = []
rating = bert(data)
#h.append(hidden_reps)
#cls.append(cls_head)
#print(h)
#print(len(cls_head[0]))
#final = nn.Linear(len(cls_head[0]), 1)
#rating = final(cls_head)
print(rating)
