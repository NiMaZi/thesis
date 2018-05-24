import json
import boto3

def get_bucket(bucket_name):
	s3=boto3.resource("s3")
	myBucket=s3.Bucket(bucket_name)
	return myBucket

def load_sups(path):
	f=open(path,'r')
	cc2vid=json.load(f)
	f.close()
	return cc2vid