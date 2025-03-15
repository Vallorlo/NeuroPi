from django.shortcuts import render
from NeuroPi.settings import TRIAL_DIR
from .models import get_users, process_eeg_data
from pathlib import Path  



def eeg_view(request):
    users = get_users(TRIAL_DIR)
    selected_user = request.GET.get("user", "") 

    print(request.GET)  
    print(request.GET.get("user"))  
    print(request.GET.get("gendy"))
    print(users)
    print(get_users(TRIAL_DIR)) 

    plots = users
    if selected_user:
        user_folder = f"{TRIAL_DIR}/trial_{selected_user}"
        plots = process_eeg_data(user_folder)
        print(plots)
        print(selected_user)

    return render(request, 'plot.html', {"users": users, "selected_user": selected_user, "plots": plots})
