import boto3

s3 = boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')
volume=10000
# volume=12935

for i in range(11804,5000+volume):
	myBucket.download_file("yalun/annotated_papers/abs"+str(i)+".csv","/home/ubuntu/thesiswork/kdata/updated/abs"+str(i)+".csv")
	myBucket.download_file("yalun/annotated_papers/body"+str(i)+".csv","/home/ubuntu/thesiswork/kdata/updated/body"+str(i)+".csv")
	# myBucket.download_file("yalun/annotated_papers/title"+str(i)+".csv","/home/ubuntu/thesiswork/kdata/updated/title"+str(i)+".csv")
	# myBucket.download_file("yalun/kdata/keywords"+str(i)+".csv","/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv")