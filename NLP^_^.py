import numpy as np
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Embedding,LSTM,SpatialDropout1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import re 

df =pd.read_csv("Sentiment.csv")
df=df[['text','sentiment']]

df=df[df.sentiment !="Neutral"]
df['text']=df['text'].apply(lambda x: x.lower())
df['text']=df['text'].apply(lambda x: re.sub(r'[^a-zA-z0-9\s]','',x))


for ind,row in df.iterrows():
    row[0]=row[0].replace('rt',' ')
    
maxfeature =2000
tokenizer= Tokenizer(num_words=maxfeature,split=' ')
tokenizer.fit_on_texts(df['text'].values)
x= tokenizer.texts_to_sequences(df['text'].values)
x=pad_sequences(x)
print(x[0])


embed_dim =128
lstm_out=196
model= Sequential()
model.add(Embedding(maxfeature, embed_dim, input_length=x.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,'softmax'))
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
y=pd.get_dummies(df['sentiment']).values



x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)

model.fit(x_train,y_train,epochs=10,batch_size=32)

