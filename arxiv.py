import os
import subprocess

for i in range(21,32):
	subprocess.call(['s3cmd','get','--requester-pays','s3://arxiv/pdf/arXiv_pdf_1805_0'+str(i)+'.tar'])
	subprocess.call(['tar','-xf','arXiv_pdf_1805_0'+str(i)+'.tar'])