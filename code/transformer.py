import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAtt(nn.Module):
	def __init__(self, n_head=2, embed_size=16, d_head=2 ):
		super(MultiHeadAtt, self).__init__()
		self.n_head = n_head
		self.d_model = embed_size
		self.d_head = d_head

		self.q_net = nn.Linear(embed_size, n_head*d_head, bias=False)
		self.kv_net = nn.Linear(embed_size, 2*n_head*d_head, bias=False)

		self.o_net = nn.Linear(n_head*d_head, embed_size, bias=False)

		self.scale = 1/(d_head ** 0.5)
		self.drop = nn.Dropout(p=0.2)

	def forward(self, h):
		head_q = self.q_net(h)
		head_k, head_v = torch.chunk(self.kv_net(h), 2, -1)
		
		head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
		head_k = head_k.view(h.size(0), h.size(1), self.n_head, self.d_head)
		head_v = head_v.view(h.size(0), h.size(1), self.n_head, self.d_head)

		att_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
		att_score.mul_(self.scale)

		att_prob = F.softmax(att_score, dim=2)

		att_vec = torch.einsum('ijbn, jbnd->ibnd', (att_prob, head_v))
		att_vec = att_vec.contiguous().view(att_vec.size(0), att_vec.size(1), self.n_head*self.d_head)

		att_out = self.o_net(att_vec)
		output = h+att_out
	
		return output

class PositionWiseFeedForward(nn.Module):
	def __init__( self, embed_size=16, ffn_dim=64 ):
		super(PositionWiseFeedForward, self).__init__()
		self.w1 = nn.Conv1d(embed_size, ffn_dim, 1)
		self.w2 = nn.Conv1d(embed_size, ffn_dim, 1)
	

	def forward(self, x):
		output = x.transpose(1, 2)
		output = self.w2(F.relu(self.w1(output)))
		output = output.transpose(1, 2)

		output = x + output
		return output
