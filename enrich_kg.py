
# coding: utf-8

# In[10]:


import os
import json
import numpy as np
import csv
import subprocess
from scipy.sparse import lil_matrix,csc_matrix,load_npz,save_npz


# In[11]:


homedir=os.environ['HOME']


# In[12]:


KG_raw=load_npz(homedir+"/results/ontology/KG_raw.npz")
KG=KG_raw.tolil()


# In[13]:


f=open(homedir+"/results/ontology/ConCode2Vid.json",'r')
cc2vid=json.load(f)
f.close()


# In[14]:


with open(homedir+"/thesiswork/source/NCIT.csv",'r',encoding='utf-8') as csvfile:
    reader=csv.reader(csvfile)
    total_enrichment=0
    for idx,item in enumerate(reader):
        if item[0]=='Class ID':
            continue
        if item[3] or item[15]:
            self_code=item[38]
            f=open(homedir+"/temp/enrich.txt",'w',encoding='utf-8')
            f.write(item[3]+"\n"+item[15])
            f.close()
            subprocess.call([
                'java',
                '-jar',homedir+'/ner/NobleJar/NobleCoder-1.0.jar',
                '-terminology','NCI_Thesaurus',
                '-input',homedir+"/temp/enrich.txt",
                '-output',homedir+"/temp/enrich",
                '-search','best-match',
                '-selectBestCandidates',
#                 '-stripSmallWords',
#                 '-stripCommonWords',
#                 '-acronymExpansion',
            ])
            f=open(homedir+"/temp/enrich/RESULTS.tsv",'r',encoding='utf-8')
            tmp=f.read().split('\n')[1:-1]
            tmp_list=[]
            for t in tmp:
                tmp_list.append(t.split('\t')[2])
            new_count=0
            for code in tmp_list:
                if code==self_code:
                    continue
                if KG[cc2vid[self_code]+1,cc2vid[code]+1]==0.0:
                    KG[cc2vid[self_code]+1,cc2vid[code]+1]=1.0
                    KG[cc2vid[code]+1,cc2vid[self_code]+1]=1.0
                    new_count+=1
            total_enrichment+=2*new_count
#             print("Enrich KG by %d new links."%(2*new_count))
#     print("Totally enriched by %d new links."%(total_enrichment))


# In[15]:


KG_out=KG.tocsc()
save_npz(homedir+"/results/ontology/KG_enriched_def.npz",KG_out)

