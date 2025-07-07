from django.shortcuts import redirect
from django.urls import resolve, reverse
from django.contrib import messages
from django.http import JsonResponse

class RoleBasedAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Get the current URL path
        path = request.path_info.lstrip('/')

        # Define accessible paths for each role
        guest_paths = ['', 'about', 'accounts/login', 'accounts/register', 'accounts/logout', 'admin', 'static', 'media']
        user_paths = guest_paths + ['trials', 'accounts/profile']
        trainer_paths = user_paths + ['plot', 'preprocessor', 'pi_main', 'motor_imagery', 'bci']

        # Skip middleware for static files and API endpoints that don't require auth
        if any(path.startswith(p) for p in ['static/', 'media/', 'admin/']):
            response = self.get_response(request)
            return response

        # Handle unauthenticated users
        if not request.user.is_authenticated:
            # Allow access to guest paths
            if not any(path.startswith(p) for p in guest_paths):
                # For AJAX requests, return JSON response
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'error': 'Authentication required',
                        'message': 'Please log in to access this feature.',
                        'redirect': '/accounts/login/'
                    }, status=401)

                messages.info(request,
                    '🔐 Please log in to access this feature. '
                    'Join NeuroPi to unlock powerful EEG analysis tools!')
                return redirect('login')
        else:
            # Handle authenticated users with role-based access
            try:
                user_role = request.user.profile.role
            except:
                # Create profile if it doesn't exist
                from .models import UserProfile
                UserProfile.objects.get_or_create(user=request.user)
                user_role = 'user'

            # Check access based on role
            if user_role == 'guest' and not any(path.startswith(p) for p in guest_paths):
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'error': 'Access denied',
                        'message': 'Please register to access this feature.',
                        'redirect': '/accounts/register/'
                    }, status=403)

                messages.warning(request,
                    '📝 Registration required! Please complete your registration to access trials and data collection features.')
                return redirect('register')

            elif user_role == 'user' and not any(path.startswith(p) for p in user_paths):
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'error': 'Insufficient privileges',
                        'message': 'Trainer access required for this feature.',
                        'redirect': '/accounts/profile/'
                    }, status=403)

                messages.warning(request,
                    '🎓 Trainer privileges required! Upgrade your role in your profile to access advanced features like data visualization, preprocessing, and neural network training.')
                return redirect('profile')

        # Process the request
        response = self.get_response(request)
        return response
