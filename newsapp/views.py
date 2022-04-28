from tkinter import Entry
from django.http import HttpResponse
from django.shortcuts import render
import requests
import json
import pandas as pd
from .models import historical_database
import csv


def home(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=general&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def business(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def entertainment(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def general(request):
	news_api_request = requests.get('https://newsapi.org/v2/top-headlines/sources?apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	# df = pd.DataFrame.from_dict(api['sources'])
	# df = df.dropna(subset = ['source'])
	# df = df[['source','author','title','description','publishedAt','content']]
	# entries = []
	# for e in df.T.to_dict().values():
	# 	entries.append(historical_database(**e))
	# historical_database.objects.bulk_create(entries)
	return render(request,'topworld.html',{'api':api})

def health(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def science(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=science&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def sports(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=sports&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def technology(request):
	news_api_request = requests.get('http://newsapi.org/v2/top-headlines?country=in&category=technology&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
	api = json.loads(news_api_request.content)
	df = pd.DataFrame.from_dict(api['articles'])
	df = df.dropna(subset = ['source'])
	df = df[['source','author','title','description','publishedAt','content']]
	entries = []
	for e in df.T.to_dict().values():
		entries.append(historical_database(**e))
	historical_database.objects.bulk_create(entries)
	return render(request,'home.html',{'api':api})

def search(request):
	if request.method == "POST":
		query = request.POST['search']
		print("request recieved",query)
		query = str(query)
		news_api_request = requests.get('http://newsapi.org/v2/top-headlines?q='+query+'&apiKey=62de380cdfdc480683ffb6d2624e5ce7')
		api = json.loads(news_api_request.content)
		df = pd.DataFrame.from_dict(api['articles'])
		df = df.dropna(subset = ['source'])
		df = df[['source','author','title','description','publishedAt','content']]
		entries = []
		for e in df.T.to_dict().values():
			entries.append(historical_database(**e))
		historical_database.objects.bulk_create(entries)
		return render(request,'home.html',{'api':api})