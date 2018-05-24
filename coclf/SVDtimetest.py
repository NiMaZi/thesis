import sys
import pickle
import random
import numpy as np
from time import time
from sklearn import svm

volume=int(sys.argv[1])
g_ratio=float(sys.argv[2])
nu=float(sys.argv[3])

f=open("/home/ubuntu/results/coclf/svd_trainlist.pkl","rb")
training_list=pickle.load(f)
f.close()

sample=random.sample(training_list,volume)

# print(sample)

# clf_linear=svm.OneClassSVM(kernel='linear')
clf_rbf=svm.OneClassSVM(kernel='rbf',gamma=(1.0/38.0)*g_ratio,nu=nu)

n_training=np.array(sample)

# start=time()

# clf_linear.fit(n_training)

# end=time()

# print(end-start)

start=time()

clf_rbf.fit(n_training)

end=time()

print(end-start)

# f=open("/home/ubuntu/results/coclf/svd_linear.pkl","wb")
# pickle.dump(clf_linear,f)
# f.close()

f=open("/home/ubuntu/results/coclf/svd_rbf_default.pkl","wb")
pickle.dump(clf_rbf,f)
f.close()