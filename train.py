import utils.util as util
from models.bownn import BOWNN


def main():
	myModel=BOWNN(40)
	d=myModel.get_d()
	e=util.bucket()
	print(d,e)

if __name__=='__main__':
	main()