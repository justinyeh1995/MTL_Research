#!/bin/sh

gdown https://drive.google.com/u/1/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
gunzip -c GoogleNews-vectors-negative300.bin.gz > model.bin
rm GoogleNews-vectors-negative300.bin.gz 

wget http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.pkl.bz2
bunzip2 wget http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.pkl.bz2
rm jawiki_20180420_300d.pkl.bz2

