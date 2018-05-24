import sys
import math
import pickle
import random
import numpy as np
from sklearn import svm

def chunks(arr, m):
	n=int(math.ceil(len(arr)/float(m)))
	return [arr[i:i+n] for i in range(0,len(arr),n)]

chunk_num=int(sys.argv[1])
volume=int(sys.argv[2])

f=open("/home/ubuntu/results/svm/saliency_samples.pickle","rb")
sample_all=pickle.load(f)
f.close()

samples=random.sample(sample_all,volume)

sample_list=chunks(samples,chunk_num)

order=[i for i in range(0,chunk_num)]

random.shuffle(order)

# linear_error_sum=0.0
rbf_error_sum=0.0
sigmoid_error_sum=0.0

for i in range(0,chunk_num):

	print("cross validation: round "+str(i)+".\n")

	v_index=order[i]
	order.remove(v_index)
	order.append(v_index)
	training_set=sample_list[order[0]]
	for i in range(1,chunk_num-1):
		training_set.extend(sample_list[order[i]])
	validating_set=sample_list[order[chunk_num-1]]

	print("building training & validating set.\n")

	n_training=np.array(training_set)
	n_validating=np.array(validating_set)
	n_training_X=n_training[:,0:3]
	n_training_y=n_training[:,3]
	n_validating_X=n_validating[:,0:3]
	n_validating_y=list(n_validating[:,3])

	# print("training SVM with linear kernel.\n")

	# clf_linear=svm.SVC(kernel='linear')
	# clf_linear.fit(n_training_X,n_training_y)
	# n_predicted_y=list(clf_linear.predict(n_validating_X))
	# error_count=0
	# for j in range(0,len(n_validating_y)):
	# 	if not n_validating_y[j]==n_predicted_y[j]:
	# 		error_count+=1
	# linear_error_sum+=float(error_count)/float(len(n_validating_y))

	print("training SVM with rbf kernel.\n")

	clf_rbf=svm.SVC(kernel='rbf',class_weight={1:1,0:31})
	clf_rbf.fit(n_training_X,n_training_y)
	n_predicted_y=list(clf_rbf.predict(n_validating_X))
	error_count=0
	for j in range(0,len(n_validating_y)):
		if not n_validating_y[j]==n_predicted_y[j]:
			error_count+=1
	rbf_error_sum+=float(error_count)/float(len(n_validating_y))

	print("training SVM with sigmoid kernel.\n")

	clf_sigmoid=svm.SVC(kernel='sigmoid',class_weight={1:1,0:31})
	clf_sigmoid.fit(n_training_X,n_training_y)
	n_predicted_y=list(clf_sigmoid.predict(n_validating_X))
	error_count=0
	for j in range(0,len(n_validating_y)):
		if not n_validating_y[j]==n_predicted_y[j]:
			error_count+=1
	sigmoid_error_sum+=float(error_count)/float(len(n_validating_y))

# linear_error_sum/=float(chunk_num)
rbf_error_sum/=float(chunk_num)
sigmoid_error_sum/=float(chunk_num)

print(rbf_error_sum,sigmoid_error_sum)