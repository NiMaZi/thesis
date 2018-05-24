import sys
import math
import pickle
import random
import numpy as np
from sklearn import svm

def chunks(arr, m):
	n=int(math.ceil(len(arr)/float(m)))
	return [arr[i:i+n] for i in range(0,len(arr),n)]

f=open("/Users/yalunzheng/Documents/BioPre/saliency/local/trainlist.pkl","rb")
train_list=pickle.load(f)
f.close()

f=open("/Users/yalunzheng/Documents/BioPre/saliency/local/testlist.pkl","rb")
test_list=pickle.load(f)
f.close()

chunk_num=int(sys.argv[1])

sample_list=chunks(train_list,chunk_num)

order=[i for i in range(0,chunk_num)]

random.shuffle(order)

linear_error_sum=0.0
rbfg1_error_sum=0.0
rbfg10_error_sum=0.0
rbfg100_error_sum=0.0
rbfg1000_error_sum=0.0

for i in range(0,chunk_num):

	print("cross validation: round "+str(i)+".\n")

	v_index=order[i]
	v_order=list(order)
	v_order.remove(v_index)
	v_order.append(v_index)
	training_set=sample_list[v_order[0]]
	for i in range(1,chunk_num-1):
		training_set.extend(sample_list[v_order[i]])
	validating_set=sample_list[v_order[chunk_num-1]]

	print("\tbuilding training & validating set.\n")

	n_training=np.array(training_set)
	n_validating=np.array(validating_set)
	n_training_X=n_training[:,0:6]
	n_training_y=n_training[:,6]
	n_validating_X=n_validating[:,0:6]
	n_validating_y=list(n_validating[:,6])

	print("\ttraining SVM with linear kernel.\n")

	clf_linear=svm.LinearSVC()
	clf_linear.fit(n_training_X,n_training_y)
	n_predicted_y=list(clf_linear.predict(n_validating_X))
	error_count=0
	for j in range(0,len(n_validating_y)):
		if not n_validating_y[j]==n_predicted_y[j]:
			error_count+=1
	linear_error_sum+=float(error_count)/float(len(n_validating_y))

	print("\ttraining SVM with rbf kernel, gamma 1.\n")

	clf_rbfg1=svm.SVC(kernel='rbf',gamma=(1.0/6.0)*1.0)
	clf_rbfg1.fit(n_training_X,n_training_y)
	n_predicted_y=list(clf_rbfg1.predict(n_validating_X))
	error_count=0
	for j in range(0,len(n_validating_y)):
		if not n_validating_y[j]==n_predicted_y[j]:
			error_count+=1
	rbfg1_error_sum+=float(error_count)/float(len(n_validating_y))

	# print("\ttraining SVM with rbf kernel, gamma /10.\n")

	# clf_rbfg10=svm.SVC(kernel='rbf',gamma=(1.0/60.0))
	# clf_rbfg10.fit(n_training_X,n_training_y)
	# n_predicted_y=list(clf_rbfg10.predict(n_validating_X))
	# error_count=0
	# for j in range(0,len(n_validating_y)):
	# 	if not n_validating_y[j]==n_predicted_y[j]:
	# 		error_count+=1
	# rbfg10_error_sum+=float(error_count)/float(len(n_validating_y))

	# print("\ttraining SVM with rbf kernel, gamma /100.\n")

	# clf_rbfg100=svm.SVC(kernel='rbf',gamma=(1.0/600.0))
	# clf_rbfg100.fit(n_training_X,n_training_y)
	# n_predicted_y=list(clf_rbfg100.predict(n_validating_X))
	# error_count=0
	# for j in range(0,len(n_validating_y)):
	# 	if not n_validating_y[j]==n_predicted_y[j]:
	# 		error_count+=1
	# rbfg100_error_sum+=float(error_count)/float(len(n_validating_y))

	# print("\ttraining SVM with rbf kernel, gamma /1000.\n")

	# clf_rbfg1000=svm.SVC(kernel='rbf',gamma=(1.0/6000.0))
	# clf_rbfg1000.fit(n_training_X,n_training_y)
	# n_predicted_y=list(clf_rbfg1000.predict(n_validating_X))
	# error_count=0
	# for j in range(0,len(n_validating_y)):
	# 	if not n_validating_y[j]==n_predicted_y[j]:
	# 		error_count+=1
	# rbfg1000_error_sum+=float(error_count)/float(len(n_validating_y))

linear_error_sum/=float(chunk_num)
rbfg1_error_sum/=float(chunk_num)
rbfg10_error_sum/=float(chunk_num)
rbfg100_error_sum/=float(chunk_num)
rbfg1000_error_sum/=float(chunk_num)

print(linear_error_sum,rbfg1_error_sum,rbfg10_error_sum,rbfg100_error_sum,rbfg1000_error_sum)