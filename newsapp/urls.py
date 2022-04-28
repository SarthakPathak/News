from django.urls import path
from newsapp import views

urlpatterns = [
    path('home',views.home),
    path('business',views.business,name='business'),
    path('entertainment',views.entertainment,name='entertainment'),
    path('general',views.general,name='general'),
    path('health',views.health,name='health'),
    path('science',views.science,name='science'),
    path('sports',views.sports,name='sports'),
    path('technology',views.technology,name='technology'),
    path('search',views.search,name='search'),
]
