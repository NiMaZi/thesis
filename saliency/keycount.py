import sys
import csv

volume=int(sys.argv[1])

count=0

for i in range(0,volume):
	f=open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".txt.mentions","r",newline='',encoding='utf-8')
	reader = csv.reader(f)
	c=0
	for item in reader:
		c+=1
	f.close()
	if c>1:
		count+=1
		continue
	f=open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".txt.mentions","r",newline='',encoding='utf-8')
	reader = csv.reader(f)
	c=0
	for item in reader:
		c+=1
	f.close()
	if c>1:
		count+=1

print(count)