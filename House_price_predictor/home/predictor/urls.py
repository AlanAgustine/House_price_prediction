# urls.py

from django.urls import path
from . import views

app_name = 'predictor'
urlpatterns = [
    # ... your other views and URLs ...
    # path('predict_price/', views.house_price_prediction, name='house_price_prediction'),
    # path("home", views.homepage, name="home"),
    # path("register", views.register_request, name="register"),
    path("", views.IndexView, name="index"),
    path("base", views.BaseView, name="base"),
    path('predict/', views.predict_price, name='predict_price'),
]
