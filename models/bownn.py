from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Concatenate,Input,Dropout,BatchNormalization

class BOWNN:
	def __init__(self,input_size,hidden_size,output_size):
		self.droprate=0.5
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size

	def build_model(self):
		model=Sequential()
		model.add(Dense(self.hidden_size,input_shape=(self.input_size,),activation='relu'))
		model.add(Dropout(self.droprate))
		model.add(BatchNormalization())
		model.add(Dense(self.output_size,activation='relu'))
		model.compile(optimizer='nadam',loss='binary_crossentropy')
		self.model=model

	def load_model(self,path):
		self.model=load_model(path)

	def save_model(self,path):
		self.model.save(path)

class BOWNN_author(BOWNN):
	def __init__(self,input_size,author_size,hidden_size,output_size):
		self.droprate=0.5
		self.input_size=input_size
		self.author_size=author_size
		self.hidden_size=hidden_size
		self.output_size=output_size

	def build_model(self):
		in_1=Input(shape=(self.input_size,))
		in_2=Input(shape=(self.author_size,))
		b1=BatchNormalization()(in_1)
		b2=BatchNormalization()(in_2)
		d1=Dropout(self.droprate)(b1)
		d2=Dropout(self.droprate)(b2)
		x1=Dense(self.hidden_size,activation='relu')(d1)
		x2=Dense(self.hidden_size,activation='relu')(d2)
		concat=Concatenate()([x1,x2])
		hidden=Dense(self.hidden_size,activation='relu')(concat)
		out1=Dense(self.output_size,activation='relu')(hidden)
		model=Model(inputs=[in_1,in_2], outputs=[out1])
		model.compile(optimizer='Nadam',loss='binary_crossentropy')
		self.model=model