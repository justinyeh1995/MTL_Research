import argparse

def process_command():
	parser = argparse.ArgumentParser(prog='Training', description='Arguments')
	parser.add_argument('--gpu', '-g', default=0, help='-1=cpu, 0, 1,...= gpt', type=int)
	parser.add_argument('--epoch', '-epoch', default=300, type=int)
	parser.add_argument('--batch', '-batch', default=32, help='batch size', type=int)
	parser.add_argument('--sample_size', '-sample', default=30, type=int)
	parser.add_argument('--lang', '-lang', default='en', help='en=English, jp=Japanese')
	parser.add_argument('--task', '-task', default='reg', help='reg=Regression, rank=Ranking')
	parser.add_argument('--train', '-train', default='../data/imdb/', help='path of training data')
	parser.add_argument('--encoder', '-train', default='BERT', help='path of training data')
	#parser.add_argument('--predict', '-predict', default='../data/test.csv', help='path of predicted data')

	return parser.parse_args()
