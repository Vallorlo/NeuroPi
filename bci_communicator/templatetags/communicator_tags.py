from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()


@register.filter
def jsonify(value):
    """Convert Python object to JSON for JavaScript"""
    return mark_safe(json.dumps(value))


@register.inclusion_tag('bci_communicator/includes/letter_grid.html')
def letter_grid(letters, side, current_selection=None):
    """Render letter grid component"""
    return {
        'letters': letters,
        'side': side,
        'current_selection': current_selection
    }


@register.inclusion_tag('bci_communicator/includes/prediction_bar.html')
def prediction_bar(label, confidence, threshold=0.7):
    """Render prediction confidence bar"""
    return {
        'label': label,
        'confidence': confidence,
        'threshold': threshold,
        'high_confidence': confidence >= threshold
    }


@register.filter
def get_class_name(predicted_class):
    """Convert motor imagery class number to name"""
    class_names = {
        0: 'Right Hand',
        1: 'Left Hand', 
        2: 'Feet',
        3: 'Rest'
    }
    return class_names.get(predicted_class, 'Unknown')


@register.filter
def format_confidence(confidence):
    """Format confidence as percentage"""
    if confidence is None:
        return "N/A"
    return f"{confidence * 100:.1f}%"


@register.filter
def time_since_event(timestamp):
    """Get time elapsed since event"""
    from django.utils import timezone
    now = timezone.now()
    diff = now - timestamp
    
    if diff.seconds < 60:
        return f"{diff.seconds}s ago"
    elif diff.seconds < 3600:
        return f"{diff.seconds // 60}m ago"
    else:
        return f"{diff.seconds // 3600}h ago"