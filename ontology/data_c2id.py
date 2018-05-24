import json

f=open("/home/ubuntu/results_new/ontology/wordlist_id.json",'r',encoding='utf-8')
wlid=json.load(f)
f.close()

c2id={}

for i in range(0,len(wlid)):
	c2id[wlid[i]]=i

f=open("/home/ubuntu/results_new/ontology/c2id.json",'w',encoding='utf-8')
json.dump(c2id,f)
f.close()