import json
import jsonlines
import pickle

count=0
keyterms_count=0
keywords_count=0
keywords_list=[]

with jsonlines.open("/home/ubuntu/pubmed_2017-05.jsonl") as reader:
	for item in reader:
		# abstract=item["abstract"]
		# title=item["title"]
		# body=item["body"]
		keywords=item["keywords"]
		if not keywords:
			continue
		keywords_list.append(keywords)
		# f=open("/home/ubuntu/thesiswork/data2/abs"+str(count)+".txt","w",encoding='utf-8')
		# f.write(abstract)
		# f.close()
		# f=open("/home/ubuntu/thesiswork/data2/body"+str(count)+".txt","w",encoding='utf-8')
		# f.write(str(body))
		# f.close()
		# f=open("/home/ubuntu/thesiswork/data2/title"+str(count)+".txt","w",encoding='utf-8')
		# f.write(title)
		# f.close()
		# f=open("/home/ubuntu/thesiswork/data2/keywords"+str(count)+".txt","wb")
		
		# f.close()
		# g=open("/home/ubuntu/thesiswork/kdata/title"+str(count)+".txt","w",encoding='utf-8')
		# keylist=item.keys()
		
		# authors=item["authors"]
		
		# units=item["units"]
		# keyterms=item["keyterms"]
		# ids=item["ids"]
		
		# date=item["date"]
		# _id=item["_id"]
		# if str(keyterms)!="None":
		# 	keyterms_count=keyterms_count+1
		# 	# print(keyterms)
		
			# keywords_count=keywords_count+1
			# print(keywords)
		# print(str(keywords))
		# print(title)
		# f.write(str(keywords))
		# f.close()
		# g.write(title)
		# g.close()
		# if count==177:
		# 	print(json.dumps(item,sort_keys=False,indent=2,separators=(',',':')))
		# 	# print(str(body))
		# 	break
		count=count+1
		# if count>5:
		# 	break
	print(count)
	# print(count,keywords_count,keyterms_count)

f=open("/home/ubuntu/results/saliency/keywords_list","wb")
pickle.dump(keywords_list,f)
f.close()