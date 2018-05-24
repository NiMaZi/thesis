import sys
import pickle
import math
import random
import numpy as np
from sklearn import svm

def chunks(arr, m):
	n=int(math.ceil(len(arr)/float(m)))
	return [arr[i:i+n] for i in range(0,len(arr),n)]

chunk_num=int(sys.argv[1])

f=open("/home/ubuntu/results/svm/samples.pickle","rb")
samples=pickle.load(f)
f.close()

sample_list=chunks(samples,chunk_num)

order=[i for i in range(0,chunk_num)]

random.shuffle(order)

test_set=np.array(sample_list[order[0]])
train_set=sample_list[order[1]]
for i in range(2,chunk_num):
	train_set.extend(sample_list[order[i]])
train_set=np.array(train_set)

clf=svm.OneClassSVM(nu=0.1,kernel="rbf",gamma=0.1)
clf.fit(train_set)

pred_train=clf.predict(train_set)
error_train=float(pred_train[pred_train==-1].size)/float(pred_train.size)

pred_test=clf.predict(test_set)
error_test=float(pred_test[pred_test==-1].size)/float(pred_test.size)

print(error_train)
print(error_test)