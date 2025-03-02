from django.urls import path

from . import views



urlpatterns = [
    path("", views.get_name, name="index"),
    path('user', views.user, name="user"),
    path('test', views.test, name="test"),

]

