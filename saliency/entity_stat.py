import pickle

f=open("/home/ubuntu/results/saliency/w_count.pkl","rb")
w=pickle.load(f)
f.close()
f=open("/home/ubuntu/results/saliency/wf_count.pkl","rb")
wf=pickle.load(f)
f.close()
f=open("/home/ubuntu/results/saliency/pwf_count.pkl","rb")
pwf=pickle.load(f)
f.close()
f=open("/home/ubuntu/results/saliency/tpwf_count.pkl","rb")
tpwf=pickle.load(f)
f.close()
f=open("/home/ubuntu/results/saliency/fpwf_count.pkl","rb")
fpwfw=pickle.load(f)
f.close()

sw=list(w.keys())
# sw=sorted(w,key=w.get)
# swf=sorted(wf,key=wf.get)
# spwf=sorted(pwf,key=pwf.get)
# stpwf=sorted(tpwf,key=tpwf.get)
# sfpwf=sorted(fpwf,key=fpwf.get)

estat={}

for k in sw:
	try:
		p=pwf[k]
	except:
		p=0.0
	try:
		tp=tpwf[k]
	except:
		tp=0.0
	try:
		fp=fpwf[k]
	except:
		fp=0.0
	print(k,w[k],wf[k],p,tp,fp)