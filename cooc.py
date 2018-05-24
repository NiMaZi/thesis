import sys
import csv
import pickle

freq_threshold=float(sys.argv[1])
test_volume=int(sys.argv[2])
entity_dict={}

f=open("/home/ubuntu/results/collectiveDict.pickle","rb")
entity_dict=pickle.load(f)
f.close()

total_hit_rate=0.0
total_error_rate=0.0
miss_count=0
invalid_count=0

for i in range(2000,2000+test_volume):
	prediction=set([])
	mention=set([])
	with open("/home/ubuntu/thesiswork/data/abs"+str(i)+".txt.mentions","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			if item[2] in entity_dict:
				for key in entity_dict[item[2]].keys():
					if entity_dict[item[2]][key]>freq_threshold:
						prediction.add(key)
	if not prediction:
		print("can't make prediction for article "+str(i)+".\n")
		miss_count=miss_count+1
		continue
	with open("/home/ubuntu/thesiswork/data/body"+str(i)+".txt.mentions","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			if item[2] not in mention:
				mention.add(item[2])
	if not mention:
		invalid_count=invalid_count+1
		continue
	hit_rate=float(len(prediction&mention))/float(len(mention))
	error_rate=1-float(len(prediction&mention))/float(len(prediction))
	total_hit_rate=total_hit_rate+hit_rate
	total_error_rate=total_error_rate+error_rate
	print("hit rate in article "+str(i)+": "+str(hit_rate)+".\n")
	print("error rate in article "+str(i)+": "+str(error_rate)+".\n")
total_hit_rate=total_hit_rate/(test_volume-miss_count-invalid_count)
total_error_rate=total_error_rate/(test_volume-miss_count-invalid_count)
miss_count=miss_count/(test_volume-invalid_count)
print("total hit rate: "+str(total_hit_rate)+".\n")
print("total error rate: "+str(total_error_rate)+".\n")
print("dictionary miss rate: "+str(miss_count)+".\n")