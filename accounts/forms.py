from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    role = forms.ChoiceField(
        choices=[
            (UserProfile.USER, 'Regular User'),
            (UserProfile.TRAINER, 'Trainer')
        ],
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=True,
        help_text='Select your role in the system'
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'role']

    def save(self, commit=True):
        user = super(UserRegistrationForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        selected_role = self.cleaned_data['role']

        if commit:
            user.save()
            # Get the profile that was created by the signal
            user_profile = user.profile
            user_profile.role = selected_role
            user_profile.save()

        return user

class UserRoleForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['role']
        widgets = {
            'role': forms.Select(attrs={'class': 'form-select'})
        }
