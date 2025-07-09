"""
Custom middleware for BCI Communicator
"""
import logging
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
from .exceptions import BCICommunicatorError

logger = logging.getLogger(__name__)


class BCIErrorHandlingMiddleware(MiddlewareMixin):
    """Handle BCI-specific errors gracefully"""
    
    def process_exception(self, request, exception):
        if isinstance(exception, BCICommunicatorError):
            logger.error(f"BCI Communicator Error: {exception}")
            
            # Return JSON for AJAX requests
            if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': str(exception)
                }, status=500)
            
            # For regular requests, let Django handle it
            return None
        
        return None
