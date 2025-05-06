from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from .forms import UserRegistrationForm, UserRoleForm

def register(request):
    """View for user registration with role selection"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            role = form.cleaned_data.get('role')

            # Store registration data in session for confirmation page
            request.session['registration_complete'] = {
                'username': username,
                'role': role,
                'role_display': 'Trainer' if role == 'trainer' else 'User'
            }

            # Send welcome email (if email settings are configured)
            try:
                send_welcome_email(user, role)
            except Exception as e:
                print(f"Error sending welcome email: {e}")

            # Redirect to confirmation page
            return redirect('registration_complete')
    else:
        form = UserRegistrationForm()

    return render(request, 'accounts/register.html', {'form': form})

def registration_complete(request):
    """View for registration confirmation"""
    # Get registration data from session
    registration_data = request.session.get('registration_complete', {})

    # Clear session data
    if 'registration_complete' in request.session:
        del request.session['registration_complete']

    # If no registration data, redirect to home
    if not registration_data:
        return redirect('home')

    return render(request, 'accounts/registration_complete.html', registration_data)

def send_welcome_email(user, role):
    """Send welcome email to new user"""
    subject = f'Welcome to NeuroPi - Your {role.capitalize()} Account'

    if role == 'trainer':
        message = f"""
        Hello {user.username},

        Welcome to NeuroPi! Your Trainer account has been created successfully.

        As a Trainer, you have full access to all features:
        - Home and About pages
        - Trials data collection and management
        - Plot visualization tools
        - Data processing capabilities
        - Neural network training and analysis

        Get started by logging in at: http://127.0.0.1:8000/accounts/login/

        Best regards,
        The NeuroPi Team
        """
    else:
        message = f"""
        Hello {user.username},

        Welcome to NeuroPi! Your User account has been created successfully.

        As a User, you have access to:
        - Home and About pages
        - Trials data collection and management

        Get started by logging in at: http://127.0.0.1:8000/accounts/login/

        Best regards,
        The NeuroPi Team
        """

    # Send email if email settings are configured
    if hasattr(settings, 'EMAIL_HOST') and settings.EMAIL_HOST:
        send_mail(
            subject,
            message,
            'noreply@neuropi.com',
            [user.email],
            fail_silently=True,
        )

@login_required
def profile(request):
    """View for user profile"""
    if request.method == 'POST':
        form = UserRoleForm(request.POST, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        form = UserRoleForm(instance=request.user.profile)

    return render(request, 'accounts/profile.html', {'form': form})
