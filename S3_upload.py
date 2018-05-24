import boto3

s3 = boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')
volume=12935

for i in range(0,volume):
	f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv","r",encoding='utf-8')
	data=f.read()
	f.close()
	myBucket.put_object(Body=data,Key="yalun/kdata/abs"+str(i)+".csv")
	f=open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv","r",encoding='utf-8')
	data=f.read()
	f.close()
	myBucket.put_object(Body=data,Key="yalun/kdata/body"+str(i)+".csv")
	f=open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".csv","r",encoding='utf-8')
	data=f.read()
	f.close()
	myBucket.put_object(Body=data,Key="yalun/kdata/title"+str(i)+".csv")
	f=open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv","r",encoding='utf-8')
	data=f.read()
	f.close()
	myBucket.put_object(Body=data,Key="yalun/kdata/keywords"+str(i)+".csv")