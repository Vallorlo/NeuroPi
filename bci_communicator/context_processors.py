"""
Context processors for BCI Communicator
"""

def communicator_settings(request):
    """Add BCI Communicator settings to template context"""
    from django.conf import settings
    
    bci_settings = getattr(settings, 'BCI_COMMUNICATOR', {})
    
    return {
        'BCI_COMMUNICATOR_SETTINGS': bci_settings,
        'COMMUNICATOR_VERSION': '1.0.0'
    }
