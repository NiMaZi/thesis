
# coding: utf-8

# In[98]:


import boto3
s3=boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')


# In[99]:


import os
homedir=os.environ['HOME']


# In[100]:


import json
f=open(homedir+"/results/ontology/full_word_list.json",'r',encoding='utf-8')
word_list=json.load(f)[1:]
f.close()


# In[101]:


i_index={}
for w in word_list:
    i_index[w.split('#')[1]]=[]
t_freq={}
for w in word_list:
    t_freq[w.split('#')[1]]=0.0
t_count=0.0
d_count=0.0


# In[102]:


import csv
def calc(source,volume,i_index,t_freq,t_count,d_count):
    for i in range(0,volume):
        myBucket.download_file("yalun/"+source+"/abs"+str(i)+".csv",homedir+"/temp/tmpcsv.csv")
        with open(homedir+"/temp/tmpcsv.csv",'r',encoding='utf-8') as cf:
            rd=csv.reader(cf)
            for item in rd:
                if item[0]=='Mention':
                    continue
                i_index[item[1]].append(d_count)
                t_freq[item[1]]+=1.0
                t_count+=1.0
        d_count+=1.0
        myBucket.download_file("yalun/"+source+"/body"+str(i)+".csv",homedir+"/temp/tmpcsv.csv")
        with open(homedir+"/temp/tmpcsv.csv",'r',encoding='utf-8') as cf:
            rd=csv.reader(cf)
            for item in rd:
                if item[0]=='Mention':
                    continue
                i_index[item[1]].append(d_count)
                t_freq[item[1]]+=1.0
                t_count+=1.0
        d_count+=1.0
    return i_index,t_freq,t_count,d_count


# In[104]:


i_index,t_freq,t_count,d_count=calc("kdata",10000,i_index,t_freq,t_count,d_count)
i_index,t_freq,t_count,d_count=calc("annotated_papers",10000,i_index,t_freq,t_count,d_count)
i_index,t_freq,t_count,d_count=calc("annotated_papers_with_txt",10000,i_index,t_freq,t_count,d_count)
i_index,t_freq,t_count,d_count=calc("annotated_papers_with_txt_new",10000,i_index,t_freq,t_count,d_count)
i_index,t_freq,t_count,d_count=calc("annotated_papers_with_txt_new2",10000,i_index,t_freq,t_count,d_count)


# In[105]:


for k in t_freq.keys():
    t_freq[k]/=t_count


# In[106]:


import numpy as np


# In[107]:


idf={}
for k in i_index.keys():
    idf[k]=np.log(d_count/(1.0+len(set(i_index[k]))))


# In[108]:


f=open(homedir+"/results/statistics/tf_all.json",'w')
json.dump(t_freq,f)
f.close()
f=open(homedir+"/results/statistics/idf.json",'w')
json.dump(idf,f)
f.close()


# In[109]:


f=open(homedir+"/results/statistics/tf_all.json",'r')
d=f.read()
f.close()
myBucket.put_object(Body=d,Key="yalun/results/statistics/tf_all.json")
f=open(homedir+"/results/statistics/idf.json",'r')
d=f.read()
f.close()
myBucket.put_object(Body=d,Key="yalun/results/statistics/idf.json")

