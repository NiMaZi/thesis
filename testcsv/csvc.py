import os
import csv

# cf=open("/Users/yalunzheng/Documents/BioPre/testcsv/abs0.csv","a")

# f=open("/Users/yalunzheng/Documents/BioPre/testcsv/abs0.txt.mentions","r")
# _str=f.read()
# cf.write(_str)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/testcsv/abs1.txt.mentions","r")
# _str=f.read()
# cf.write(_str)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/testcsv/abs2.txt.mentions","r")
# _str=f.read()
# cf.write(_str)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/testcsv/abs3.txt.mentions","r")
# _str=f.read()
# cf.write(_str)
# f.close()

# cf.close()
file_dir="/Users/yalunzheng/Downloads/terminologies/"

term_list=[]
for file in os.listdir(file_dir):
	if file.split(".")[0]:
		print(file.split(".")[0])

# with open("/Users/yalunzheng/Documents/BioPre/testcsv/abs0.csv","r",newline='',encoding='utf-8') as csvfile:
# 	reader=csv.reader(csvfile)
# 	for item in reader:
# 		print(item)