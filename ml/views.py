from django.shortcuts import render
import pandas as pd
from newsapp.models import historical_database

# Libraries used for fake news detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split

# Create your views here.
def fake_news(request):
	print("Fake News")
	df = pd.DataFrame.from_records(
    historical_database.objects.all().values_list('source','author','title','description','publishedAt','content')
	)
	df.drop_duplicates(inplace=True,keep='first')
	df.reset_index(inplace=True)
	df.to_csv('cleaned_news_data.csv',"w")
	# Algorithms
	return render(request,'fake_news.html')

def Sentiment_Analysis(request):
	print("Sentiment Analysis")
	df = pd.DataFrame.from_records(
    historical_database.objects.all().values_list('source','author','title','description','publishedAt','content')
	)
	df.drop_duplicates(inplace=True,keep='first')
	df.reset_index(inplace=True)
	df.to_csv('cleaned_news_data.csv',"w")
	return render(request,"sentiment.html")