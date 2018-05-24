
# coding: utf-8

# In[23]:


import json
import numpy as np
from gensim.models import word2vec as w2v


# In[24]:


f=open("/home/ubuntu/thesiswork/source/coded_syns.json",'r')
coded_syns=json.load(f)
f.close()


# In[25]:


def load_models():
    path="/home/ubuntu/results/models/e2v_sg_140k_e200_d64.model"
    e2v_model=w2v.Word2Vec.load(path)
    f=open("/home/ubuntu/results/ontology/KG_n2v_d64.json",'r')
    n2v_model=json.load(f)
    f.close()
    return e2v_model,n2v_model

e2v_model,n2v_model=load_models()

def load_sups():
    f=open("/home/ubuntu/results/ontology/c2id.json",'r')
    c2id=json.load(f)
    f.close()
    prefix='http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'
    return c2id,prefix

c2id,prefix=load_sups()


# In[26]:


f=open("/home/ubuntu/results/ontology/full_word_list.json",'r')
word_list=json.load(f)[1:]
f.close()


# In[27]:


def get_emb(_code):
    e_vec=list(e2v_model.wv[_code])
    n_vec=n2v_model[str(c2id[prefix+_code])]
    return e_vec+n_vec

def get_embe(_code):
    e_vec=list(e2v_model.wv[_code])
    # n_vec=n2v_model[str(c2id[prefix+_code])]
    return e_vec

def get_embn(_code):
    # e_vec=list(e2v_model.wv[_code])
    n_vec=n2v_model[str(c2id[prefix+_code])]
    return n_vec


# In[28]:


from scipy.spatial.distance import cosine


# In[ ]:

f=open("/home/ubuntu/results/logs/emb_evaluation_sg.txt","w")

avg_pos_syn0=0.0
avg_pos_syn1=0.0
for i,syn in enumerate(coded_syns):
    res_dict_syn0={}
    res_dict_syn1={}
    emb_syn0=np.array(get_embe(syn[0]))
    emb_syn1=np.array(get_embe(syn[1]))
    for w in word_list:
        _w=w.split('#')[1]
        emb_w=np.array(get_embe(_w))
        dist_syn0=cosine(emb_syn0,emb_w)
        dist_syn1=cosine(emb_syn1,emb_w)
        res_dict_syn0[_w]=dist_syn0
        res_dict_syn1[_w]=dist_syn1
    cur_pos_syn0=sorted(res_dict_syn0,key=res_dict_syn0.get).index(syn[1])
    cur_pos_syn1=sorted(res_dict_syn1,key=res_dict_syn1.get).index(syn[0])
    f.write("%s,%d,%s,%d\n"%(syn[0],cur_pos_syn0,syn[1],cur_pos_syn1))
    avg_pos_syn0+=cur_pos_syn0
    avg_pos_syn1+=cur_pos_syn1
avg_pos_syn0/=len(coded_syns)
avg_pos_syn1/=len(coded_syns)


f.write("%f,%f\n"%(avg_pos_syn0,avg_pos_syn1))


# avg_pos_syn0=0.0
# avg_pos_syn1=0.0
# for i,syn in enumerate(coded_syns):
#     res_dict_syn0={}
#     res_dict_syn1={}
#     emb_syn0=np.array(get_embn(syn[0]))
#     emb_syn1=np.array(get_embn(syn[1]))
#     for w in word_list:
#         _w=w.split('#')[1]
#         emb_w=np.array(get_embn(_w))
#         dist_syn0=cosine(emb_syn0,emb_w)
#         dist_syn1=cosine(emb_syn1,emb_w)
#         res_dict_syn0[_w]=dist_syn0
#         res_dict_syn1[_w]=dist_syn1
#     cur_pos_syn0=sorted(res_dict_syn0,key=res_dict_syn0.get).index(syn[1])
#     cur_pos_syn1=sorted(res_dict_syn1,key=res_dict_syn1.get).index(syn[0])
#     f.write("%s,%d,%s,%d\n"%(syn[0],cur_pos_syn0,syn[1],cur_pos_syn1))
#     avg_pos_syn0+=cur_pos_syn0
#     avg_pos_syn1+=cur_pos_syn1
# avg_pos_syn0/=len(coded_syns)
# avg_pos_syn1/=len(coded_syns)


# f.write("%f,%f\n"%(avg_pos_syn0,avg_pos_syn1))


# avg_pos_syn0=0.0
# avg_pos_syn1=0.0
# for i,syn in enumerate(coded_syns):
#     res_dict_syn0={}
#     res_dict_syn1={}
#     emb_syn0=np.array(get_emb(syn[0]))
#     emb_syn1=np.array(get_emb(syn[1]))
#     for w in word_list:
#         _w=w.split('#')[1]
#         emb_w=np.array(get_emb(_w))
#         dist_syn0=cosine(emb_syn0,emb_w)
#         dist_syn1=cosine(emb_syn1,emb_w)
#         res_dict_syn0[_w]=dist_syn0
#         res_dict_syn1[_w]=dist_syn1
#     cur_pos_syn0=sorted(res_dict_syn0,key=res_dict_syn0.get).index(syn[1])
#     cur_pos_syn1=sorted(res_dict_syn1,key=res_dict_syn1.get).index(syn[0])
#     f.write("%s,%d,%s,%d\n"%(syn[0],cur_pos_syn0,syn[1],cur_pos_syn1))
#     avg_pos_syn0+=cur_pos_syn0
#     avg_pos_syn1+=cur_pos_syn1
# avg_pos_syn0/=len(coded_syns)
# avg_pos_syn1/=len(coded_syns)


# f.write("%f,%f\n"%(avg_pos_syn0,avg_pos_syn1))
f.close()

