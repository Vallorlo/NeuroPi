from django.shortcuts import redirect
from django.urls import resolve, reverse
from django.contrib import messages

class RoleBasedAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Get the current URL path
        path = request.path_info.lstrip('/')
        
        # Define accessible paths for each role
        guest_paths = ['', 'about', 'accounts/login', 'accounts/register', 'accounts/logout', 'admin']
        user_paths = guest_paths + ['trials']
        trainer_paths = user_paths + ['plot', 'preprocessor', 'pi_main']
        
        # Check if the user is authenticated
        if request.user.is_authenticated:
            # Get user role
            user_role = request.user.profile.role
            
            # Check access based on role
            if user_role == 'guest' and not any(path.startswith(p) for p in guest_paths):
                messages.warning(request, 'You need to register to access this page.')
                return redirect('home')
            elif user_role == 'user' and not any(path.startswith(p) for p in user_paths):
                messages.warning(request, 'You need trainer privileges to access this page.')
                return redirect('home')
        
        # Process the request
        response = self.get_response(request)
        return response
