from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name="challenges"),
    path('postState/', views.postState, name="postState"),
    path('getAgent', views.getAgent, name="getAgent"),

]
