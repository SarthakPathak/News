import pandas as pd 
import numpy as np
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from textblob import TextBlob
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score,recall_score
import seaborn as sns
import pickle

#intialize spacy word model
nlp=spacy.load('en_core_web_sm')

def read_data():
    dataset=pd.read_csv('all-data.csv',
                header=None,
                names=['Sentiment','News'], encoding='ISO-8859-1')
    return dataset

def convert_to_lower_case():
    def lower(input_text):
        return input_text.lower()
    dataset['News']=dataset['News'].apply(lower)
    
def remove_punctuation():
    def remove_punctuation_from_text(input_text):
        output_list=[word for word in input_text.split() if word.isalpha()]
        return ' '.join(output_list)    
    dataset['News']=dataset['News'].apply(remove_punctuation_from_text)
    
def correct_words():
    def correct_text(input_text):
        list_1=[str(TextBlob(word).correct()) for word in input_text.split()]
        output_text= ' '.join(list_1)
        return output_text
    dataset['News']=dataset['News'].apply(correct_text)
    
def lemmatize():
    def lematize_text(input_text):
        doc=nlp(input_text)
        lemmas=[token.lemma_ for token in doc]
        output_text=' '.join(lemmas)
        return output_text
    dataset['News']=dataset['News'].apply(lematize_text)
    
def remove_stopwords():
    def remove_stopwords_from_text(input_text):
        stopwords=spacy.lang.en.stop_words.STOP_WORDS
        output_list=[word for word in input_text.split() if word not in stopwords and not(word=='-PRON-') ]
        return ' '.join(output_list)
    dataset['News']=dataset['News'].apply(remove_stopwords_from_text)

def filter_the_neutral_news():
    return dataset[dataset['Sentiment']!='neutral']

def create_target_and_input():
    target=dataset['Sentiment'].values.tolist()
    target=[1 if sentiment=='positive' else 0 for sentiment in target]
    data=dataset['News'].values.tolist()
    return data,target

def split_train_test():
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42,stratify=target)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return x_train, x_test, y_train, y_test 

#reading the data
dataset=read_data()

#Preprocessing the text
convert_to_lower_case()
remove_punctuation()
lemmatize()
remove_stopwords()

#Preparing data for model
dataset=filter_the_neutral_news()
data, target=create_target_and_input()
x_train, x_test, y_train, y_test =split_train_test()

#setting a threshold for the number of words that we are going to use

num_words=1000 # number of words that we are going to use. It takes top 1k words with the highest frequency
tokenizer=Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data)

#tokenization
x_train_tokens=tokenizer.texts_to_sequences(x_train)
x_test_tokens=tokenizer.texts_to_sequences(x_test)

#setting a threshold for the number of words in each text
num_tokens=[len(tokens) for tokens in x_train_tokens+x_test_tokens]
num_tokens=np.array(num_tokens)
max_tokens=np.mean(num_tokens)+2*np.std(num_tokens)
max_tokens=int(max_tokens)

#padding
x_train_pad=pad_sequences(x_train_tokens,
                              maxlen=max_tokens)
x_test_pad=pad_sequences(x_test_tokens,
                         maxlen=max_tokens)

#creating model
model=Sequential()
embedding_size=50  # we will create a 50 size vector for each word.
#At the beginning we will use random word vectors and each optimization step these vectors will be  
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer')
) # this Embedding layer will take a text as an input, convert it to a vector as an output

model.add(GRU(units=16, # number of neurons 
              return_sequences=True) # if true this layer odel creates multiple outputs. If the following layer has one neuron, which means the following layer creates the output. 
)
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1,activation='sigmoid'))#with the sigmoid activation function, we receive an output between 0 and 1.

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.fit(x_train_pad,
          y_train,
          epochs=50)

result=model.evaluate(x_test_pad,
                      y_test)

filename = 'finalized_model_sentimentanalysis.sav'
pickle.dump(model,open(filename,'wb'))

#model success on the test dataset
y_test_pred=model.predict(x=x_test_pad)
y_test_pred=y_test_pred.T[0]
y_test_pred=np.array([1.0 if p>0.5 else 0.0 for p in y_test_pred])

precision_scr=precision_score(y_test, y_test_pred)
recall_scr=recall_score(y_test, y_test_pred)

print('Precision Score: {:.2f}'.format(precision_scr))
print('Recall Score: {:.2f}'.format(recall_scr))