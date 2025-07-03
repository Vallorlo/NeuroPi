"""
Custom template filters for BCI application
"""

from django import template

register = template.Library()


@register.filter
def title_case(value):
    """
    Convert to title case and replace underscores with spaces
    Usage: {{ value|title_case }}
    """
    return str(value).replace('_', ' ').title()


@register.filter
def underscore_to_space(value):
    """
    Replace underscores with spaces
    Usage: {{ value|underscore_to_space }}
    """
    return str(value).replace('_', ' ')


@register.filter
def replace(value, args):
    """
    Replace occurrences of one string with another
    Usage: {{ value|replace:"old,new" }}
    """
    if args and ',' in args:
        old, new = args.split(',', 1)
        return str(value).replace(old, new)
    return value


@register.filter
def mul(value, arg):
    """
    Multiply the value by the argument
    Usage: {{ value|mul:2 }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter
def get_item(dictionary, key):
    """
    Get an item from a dictionary by key
    Usage: {{ dict|get_item:key }}
    """
    if hasattr(dictionary, 'get'):
        return dictionary.get(key)
    return None


@register.filter
def percentage(value):
    """
    Convert decimal to percentage
    Usage: {{ 0.75|percentage }} -> 75%
    """
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "0%"


@register.filter
def duration_seconds(samples, rate=128):
    """
    Convert sample count to duration in seconds
    Usage: {{ total_samples|duration_seconds }}
    """
    try:
        return float(samples) / float(rate)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter
def format_duration(seconds):
    """
    Format duration in human readable format
    Usage: {{ seconds|format_duration }}
    """
    try:
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    except (ValueError, TypeError):
        return "0s"


@register.filter
def class_badge_color(approach):
    """
    Get Bootstrap color class for approach
    Usage: {{ approach|class_badge_color }}
    """
    color_map = {
        'motor_imagery': 'primary',
        'p300': 'info',
        'completed': 'success',
        'training': 'warning',
        'failed': 'danger',
        'running': 'success',
        'stopped': 'secondary',
        'error': 'danger'
    }
    return color_map.get(str(approach).lower(), 'secondary')


@register.filter
def status_icon(status):
    """
    Get FontAwesome icon for status
    Usage: {{ status|status_icon }}
    """
    icon_map = {
        'completed': 'fas fa-check-circle',
        'training': 'fas fa-cog fa-spin',
        'failed': 'fas fa-times-circle',
        'running': 'fas fa-play-circle',
        'stopped': 'fas fa-stop-circle',
        'error': 'fas fa-exclamation-triangle'
    }
    return icon_map.get(str(status).lower(), 'fas fa-circle')


@register.filter
def join_limit(value, arg):
    """
    Join list with separator, limiting to N items
    Usage: {{ list|join_limit:"3, " }}
    """
    try:
        if ',' in arg:
            limit_str, separator = arg.split(',', 1)
            limit = int(limit_str)
        else:
            limit = int(arg)
            separator = ', '
        
        if isinstance(value, (list, tuple)):
            limited = list(value)[:limit]
            result = separator.join(str(item) for item in limited)
            if len(value) > limit:
                result += f"{separator}..."
            return result
    except (ValueError, TypeError, AttributeError):
        pass
    return value


@register.filter
def dict_get(dictionary, key):
    """
    Get value from dictionary
    Usage: {{ my_dict|dict_get:"key_name" }}
    """
    if hasattr(dictionary, 'get'):
        return dictionary.get(key)
    return None


@register.filter
def format_size(bytes_value):
    """
    Format file size in human readable format
    Usage: {{ file_size|format_size }}
    """
    try:
        bytes_value = float(bytes_value)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    except (ValueError, TypeError):
        return "0 B"


@register.filter
def timedelta_seconds(td):
    """
    Convert timedelta to seconds
    Usage: {{ my_timedelta|timedelta_seconds }}
    """
    try:
        return td.total_seconds()
    except AttributeError:
        return 0


@register.simple_tag
def approach_display_name(approach):
    """
    Get display name for approach
    Usage: {% approach_display_name "motor_imagery" %}
    """
    display_names = {
        'motor_imagery': 'Motor Imagery',
        'p300': 'P300'
    }
    return display_names.get(approach, approach.replace('_', ' ').title())


@register.simple_tag
def status_display(status):
    """
    Get formatted status display
    Usage: {% status_display "running" %}
    """
    status_map = {
        'running': '<span class="badge bg-success">Running</span>',
        'stopped': '<span class="badge bg-secondary">Stopped</span>',
        'training': '<span class="badge bg-warning">Training</span>',
        'completed': '<span class="badge bg-success">Completed</span>',
        'failed': '<span class="badge bg-danger">Failed</span>',
        'error': '<span class="badge bg-danger">Error</span>'
    }
    return status_map.get(str(status).lower(), f'<span class="badge bg-secondary">{status}</span>')


@register.filter
def default_if_none_or_empty(value, default):
    """
    Return default if value is None, empty string, or empty list
    Usage: {{ value|default_if_none_or_empty:"Default Value" }}
    """
    if value is None or value == '' or (hasattr(value, '__len__') and len(value) == 0):
        return default
    return value