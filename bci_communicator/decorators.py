"""
Custom decorators for BCI Communicator views
"""
from functools import wraps
from django.http import JsonResponse, Http404
from django.shortcuts import get_object_or_404
from .models import CommunicationSession


def session_required(view_func):
    """Decorator to ensure session exists and belongs to user"""
    @wraps(view_func)
    def wrapper(request, session_id, *args, **kwargs):
        session = get_object_or_404(
            CommunicationSession, 
            id=session_id, 
            user=request.user
        )
        return view_func(request, session_id, session=session, *args, **kwargs)
    return wrapper


def ajax_required(view_func):
    """Decorator to ensure request is AJAX"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return JsonResponse({'error': 'AJAX required'}, status=400)
        return view_func(request, *args, **kwargs)
    return wrapper


def active_session_required(view_func):
    """Decorator to ensure session is active"""
    @wraps(view_func)
    def wrapper(request, session_id, *args, **kwargs):
        session = get_object_or_404(
            CommunicationSession, 
            id=session_id, 
            user=request.user,
            is_active=True
        )
        return view_func(request, session_id, session=session, *args, **kwargs)
    return wrapper
