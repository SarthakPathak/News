from django.urls import path
from ml import views

urlpatterns = [
    path('fake_news',views.fake_news,name='fake_news'),
    path('sentiment',views.Sentiment_Analysis,name='sentiment'),
]
