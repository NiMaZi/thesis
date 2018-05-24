import sys
import csv

volume=int(sys.argv[1])
real_volume=volume

record_abs=0.0
amb_count=0.0

for i in range(0,volume):
	print("counting on article "+str(i)+".")
	record_dict_abs={}
	try:
		with open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				entity=item[2]
				pos=item[4]
				record_abs+=1
				if pos in record_dict_abs.keys():
					record_dict_abs[pos]+=1
				else:
					record_dict_abs[pos]=1
		for key in record_dict_abs.keys():
			if record_dict_abs[key]>1:
				amb_count+=record_dict_abs[key]
	except:
		real_volume-=1
		continue

record_abs/=real_volume
amb_count/=real_volume

print(record_abs,amb_count,amb_count/record_abs)