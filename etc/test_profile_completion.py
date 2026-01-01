#!/usr/bin/env python
"""
Test script to check profile completion logic
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroPi.settings')
django.setup()

from django.contrib.auth.models import User
from user_management.models import ParticipantProfile

def test_profile_completion():
    """Test profile completion logic for all users"""
    print("=== Profile Completion Test ===\n")
    
    users = User.objects.all()
    print(f"Found {users.count()} users in the system\n")
    
    for user in users:
        print(f"User: {user.username}")
        
        try:
            profile = user.participant_profile
            print(f"  - Has participant profile: YES")
            print(f"  - Date of birth: {profile.date_of_birth}")
            print(f"  - Gender: {profile.gender}")
            print(f"  - Handedness: {profile.handedness}")
            print(f"  - Education: {profile.education_level}")
            print(f"  - Occupation: {profile.occupation}")
            print(f"  - Native language: {profile.native_language}")
            print(f"  - Profile complete: {profile.is_profile_complete}")
            print(f"  - Completion percentage: {profile.profile_completion_percentage}%")
            
        except ParticipantProfile.DoesNotExist:
            print(f"  - Has participant profile: NO")
            
        print("-" * 50)

if __name__ == "__main__":
    test_profile_completion()
