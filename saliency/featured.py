# after distanced

import sys
import pickle

f=open("/home/ubuntu/results/saliency/distanced.pkl","rb")
dis_list=pickle.load(f)
f.close()

featured_list=[]

for entry in dis_list:

	abs_list=entry['abs']
	abs_dict={}
	for item in abs_list:
		if item[1] in abs_dict:
			if item[2]<abs_dict[item[1]][0]:
				abs_dict[item[1]][0]=item[2]
			if item[2]>abs_dict[item[1]][1]:
				abs_dict[item[1]][1]=item[2]
			abs_dict[item[1]][2]+=1
		else:
			abs_dict[item[1]]=[item[2],item[2],1] #(float)min_pos, (float)max_pos, (int)count

	body_list=entry['body']
	body_dict={}
	for item in body_list:
		if item[1] in body_dict:
			if item[2]<body_dict[item[1]][0]:
				body_dict[item[1]][0]=item[2]
			if item[2]>body_dict[item[1]][1]:
				body_dict[item[1]][1]=item[2]
			body_dict[item[1]][2]+=1
		else:
			body_dict[item[1]]=[item[2],item[2],1]

	featured_list.append({'abs':abs_dict,'body':body_dict})

f=open("/home/ubuntu/results/saliency/featured.pkl","wb")
pickle.dump(featured_list,f)
f.close()