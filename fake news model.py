import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from joblib import dump,load

nltk.download()

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#filling NULL values with empty string
df=df.fillna('')
test=test.fillna('')

# We will be only using title and author name for prediction
# Creating new coolumn total concatenating title and author
df['total'] = df['title']+' '+df['author']
test['total']=test['title']+' '+test['author']
X = df.drop('label',axis=1)
y=df['label']

#Choosing vocabulary size to be 5000 and copying data to msg for further cleaning
voc_size = 5000
msg = X.copy()
msg_test = test.copy()

#Downloading stopwords 
#Stopwords are the words in any language which does not add much meaning to a sentence.
#They can safely be ignored without sacrificing the meaning of the sentence.
nltk.download('stopwords')

#We will be using Stemming here
#Stemming map words to their root forms
ps = PorterStemmer()
corpus = []

#Applying stemming and some preprocessing
for i in range(len(msg)):
  review = re.sub('[^a-zA-Z]',' ',msg['total'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)
    
#Applying stemming and some preprocessing for test data
corpus_test = []
for i in range(len(msg_test)):
  review = re.sub('[^a-zA-Z]',' ',msg_test['total'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus_test.append(review)
    
# Converting to one hot representation
onehot_rep = [one_hot(words,voc_size)for words in corpus]
onehot_rep_test = [one_hot(words,voc_size)for words in corpus_test]

#Padding Sentences to make them of same size
embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=25)
embedded_docs_test = pad_sequences(onehot_rep_test,padding='pre',maxlen=25)

# Creating a model
model = Sequential()
model.add(Embedding(voc_size,40,input_length=25))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary)

#Converting into numpy array
X_final = np.array(embedded_docs)
y_final = np.array(y)
test_final = np.array(embedded_docs_test)
X_final.shape,y_final.shape,test_final.shape

#training model
model.fit(X_final,y_final,epochs=20)

dump(model,'model_jlib_fk.joblib')

#threshold = 0.5
#y_pred = np.argmax(model.predict(), axis=-1)
#np.where(y_pred > threshold, 1,0)
y_pred = (model.predict(test_final) > 0.5).astype("int32")
final_res = pd.DataFrame()
final_res['id']=test['id']
final_res['label'] = y_pred
final_res.to_csv('final_result.csv',index=False)
final_res.head()