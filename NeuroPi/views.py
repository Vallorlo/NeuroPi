from django.http import HttpResponse
from django.shortcuts import render

def homepage(request):
    return render(request, 'home.html') #rendering what we return when we get he request



def about(request):
    return render(request,'about.html')