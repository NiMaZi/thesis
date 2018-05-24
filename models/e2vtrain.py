import json
from gensim.models import word2vec

# sentences=word2vec.LineSentence("/home/ubuntu/thesiswork/source/corpus/fullcorpus10000.txt")
sentences=word2vec.LineSentence("/home/ubuntu/thesiswork/source/corpus/corpus_all.txt")
# sentences=word2vec.LineSentence("/home/ubuntu/results/hivo_corpus.txt")

f=open("/home/ubuntu/results/statistics/tf_all.json",'r')
tf_all=json.load(f)
f.close()

tf_all_com={}
for k in tf_all.keys():
	_k=k.split("#")[1]
	tf_all_com[_k]=tf_all[k]+1e-10

# model=word2vec.Word2Vec(sg=0,size=128,window=10,min_count=0,sample=1e-3,hs=0,negative=1,sorted_vocab=1)
# model.build_vocab_from_freq(tf_all_com)
# model.train(sentences,total_examples=10000,epochs=100)
# path="/home/ubuntu/results/models/e2v_sg_5000_e100_d128.model"
# # path="/home/ubuntu/results/e2v_sg_e100.model"
# model.save(path)

# path="/home/ubuntu/results/models/e2v_sg_10000_e200_d64.model"
# model=word2vec.Word2Vec.load(path)

# ======

model=word2vec.Word2Vec(sg=0,size=64,window=10,min_count=0,sample=1e-3,hs=0,negative=5,workers=4,sorted_vocab=1,compute_loss=True)
model.build_vocab_from_freq(tf_all_com)
model.train(sentences,total_examples=140000,epochs=200)
path="/home/ubuntu/results/models/e2v_sg_140k_e200_d64.model"
# path="/home/ubuntu/results/e2v_sg_e100.model"
model.save(path)

# ======

# model=word2vec.Word2Vec(sg=0,size=32,window=10,min_count=0,sample=1e-3,hs=0,negative=1,sorted_vocab=1)
# model.build_vocab_from_freq(tf_all_com)
# model.train(sentences,total_examples=10000,epochs=100)
# path="/home/ubuntu/results/models/e2v_sg_5000_e100_d32.model"
# # path="/home/ubuntu/results/e2v_sg_e100.model"
# model.save(path)

# model=word2vec.Word2Vec(sg=1,size=128,window=10,min_count=0,sample=1e-3,hs=0,negative=1,sorted_vocab=1)
# model.build_vocab_from_freq(tf_all_com)
# model.train(sentences,total_examples=10000,epochs=100)
# path="/home/ubuntu/results/models/e2v_cbow_5000_e100_d128.model"
# # path="/home/ubuntu/results/e2v_cbow_e100.model"
# model.save(path)

# ======

model=word2vec.Word2Vec(sg=1,size=64,window=10,min_count=0,sample=1e-3,hs=0,negative=5,workers=4,sorted_vocab=1,compute_loss=True)
model.build_vocab_from_freq(tf_all_com)
model.train(sentences,total_examples=140000,epochs=200)
path="/home/ubuntu/results/models/e2v_cbow_140k_e200_d64.model"
# path="/home/ubuntu/results/e2v_cbow_e100.model"
model.save(path)

# ======

# model=word2vec.Word2Vec(sg=1,size=32,window=10,min_count=0,sample=1e-3,hs=0,negative=1,sorted_vocab=1)
# model.build_vocab_from_freq(tf_all_com)
# model.train(sentences,total_examples=10000,epochs=100)
# path="/home/ubuntu/results/models/e2v_cbow_5000_e100_d32.model"
# # path="/home/ubuntu/results/e2v_cbow_e100.model"
# model.save(path)