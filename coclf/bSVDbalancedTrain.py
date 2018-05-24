import sys
import math
import random
import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model as lm

g_ratio=float(sys.argv[1])
n_tree=int(sys.argv[2])

pos_list=[]
neg_prelist=[]

for chunk in range(0,6):
	f=open("/home/ubuntu/results/coclf/bsvd_trainlist"+str(chunk)+".pkl","rb")
	_list=pickle.load(f)
	f.close()
	for item in _list:
		if item[38]==1:
			pos_list.append(item)
		else:
			neg_prelist.append(item)

neg_list=random.sample(neg_prelist,len(pos_list))

pos_list.extend(neg_list)
train_list=list(pos_list)
random.shuffle(train_list)

print(len(train_list))

n_train=np.array(train_list)
n_train_X=n_train[:,0:38]
n_train_y=n_train[:,38]

clf_linear=svm.LinearSVC()
clf_rbf=svm.SVC(gamma=(1.0/38.0)*g_ratio)
clf_rfc=RandomForestClassifier(n_estimators=n_tree)
clf_abc=AdaBoostClassifier(n_estimators=n_tree)

clf_linear.fit(n_train_X,n_train_y)
clf_rbf.fit(n_train_X,n_train_y)
clf_rfc.fit(n_train_X,n_train_y)
clf_abc.fit(n_train_X,n_train_y)

f=open("/home/ubuntu/results/coclf/bsvd_linear.pkl","wb")
pickle.dump(clf_linear,f)
f.close()

f=open("/home/ubuntu/results/coclf/bsvd_rbf_default.pkl","wb")
pickle.dump(clf_rbf,f)
f.close()

f=open("/home/ubuntu/results/coclf/bsvd_rfc.pkl","wb")
pickle.dump(clf_rfc,f)
f.close()

f=open("/home/ubuntu/results/coclf/bsvd_abc.pkl","wb")
pickle.dump(clf_abc,f)
f.close()