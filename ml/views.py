from django.shortcuts import render
import pandas as pd
from newsapp.models import historical_database

# Create your views here.
def fake_news(request):
	print("Fake News")
	df = pd.DataFrame.from_records(
    historical_database.objects.all().values_list('source','author','title','description','publishedAt','content')
	)
	df.drop_duplicates(inplace=True,keep='first')
	df.reset_index(inplace=True)
	df.to_csv('cleaned_news_data.csv',"w")
	print("Data has been cleaned")
	return render(request,'fake_news.html')

def Sentiment_Analysis(request):
	print("Sentiment Analysis")
	df = pd.DataFrame.from_records(
    historical_database.objects.all().values_list('source','author','title','description','publishedAt','content')
	)
	df.drop_duplicates(inplace=True,keep='first')
	df.reset_index(inplace=True)
	df.to_csv('cleaned_news_data.csv',"w")
	print("Data has been cleaned")
	return render(request,"sentiment.html")