import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroPi.settings')
django.setup()

# Import models
from django.contrib.auth.models import User
from accounts.models import UserProfile

# Create or update a user with a specific role
def create_or_update_user(username, email, password, role):
    try:
        # Check if user already exists
        if User.objects.filter(username=username).exists():
            user = User.objects.get(username=username)
            try:
                # Try to get the profile
                profile = UserProfile.objects.get(user=user)
                profile.role = role
                profile.save()
            except UserProfile.DoesNotExist:
                # Create profile if it doesn't exist
                profile = UserProfile.objects.create(user=user, role=role)

            print(f"{role.capitalize()} user updated.")
        else:
            # Create new user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )

            try:
                # Try to get the profile
                profile = UserProfile.objects.get(user=user)
                profile.role = role
                profile.save()
            except UserProfile.DoesNotExist:
                # Create profile if it doesn't exist
                profile = UserProfile.objects.create(user=user, role=role)

            print(f"{role.capitalize()} user created.")
    except Exception as e:
        print(f"Error creating/updating {role} user: {e}")

if __name__ == '__main__':
    create_or_update_user('trainer', 'trainer@example.com', 'trainer123', 'trainer')
    create_or_update_user('user', 'user@example.com', 'user123', 'user')
    create_or_update_user('guest', 'guest@example.com', 'guest123', 'guest')
    print("All users created/updated successfully.")
