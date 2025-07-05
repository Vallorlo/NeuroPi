"""
Custom template filters for BCI application
"""

from django import template
from django.utils.safestring import mark_safe
import json
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



@register.filter
def mod(value, divisor):
    """
    Return the modulo of value and divisor
    Usage: {{ forloop.counter|mod:5 }}
    """
    try:
        return int(value) % int(divisor)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter
def add(value, arg):
    """
    Add the arg to the value
    Usage: {{ value|add:10 }}
    """
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        try:
            return float(value) + float(arg)
        except (ValueError, TypeError):
            return value


@register.filter
def yesno(value, choices):
    """
    Return different values based on truthiness
    Usage: {{ forloop.counter|mod:5|yesno:"primary,success,warning,danger,info" }}
    """
    if not choices:
        return value
    
    choice_list = choices.split(',')
    
    if len(choice_list) == 1:
        return choice_list[0] if value else ''
    elif len(choice_list) == 2:
        return choice_list[0] if value else choice_list[1]
    else:
        # For multiple choices, treat value as index
        try:
            index = int(value) % len(choice_list)
            return choice_list[index]
        except (ValueError, TypeError):
            return choice_list[0] if value else (choice_list[1] if len(choice_list) > 1 else '')


@register.filter
def floatformat(value, decimals=1):
    """
    Format float to specified decimal places
    Usage: {{ value|floatformat:2 }}
    """
    try:
        return f"{float(value):.{int(decimals)}f}"
    except (ValueError, TypeError):
        return value


@register.filter
def truncatechars(value, length):
    """
    Truncate string to specified length
    Usage: {{ text|truncatechars:50 }}
    """
    try:
        length = int(length)
        if len(str(value)) > length:
            return str(value)[:length-3] + '...'
        return str(value)
    except (ValueError, TypeError):
        return value


@register.filter
def timesince_short(value):
    """
    Get short timesince format
    Usage: {{ timestamp|timesince_short }}
    """
    from django.utils.timesince import timesince
    try:
        time_diff = timesince(value)
        # Get first part only (e.g., "2 hours" from "2 hours, 30 minutes")
        return time_diff.split(',')[0]
    except:
        return 'unknown'


@register.filter
def to_json(value):
    """
    Convert value to JSON string
    Usage: {{ data|to_json }}
    """
    try:
        return mark_safe(json.dumps(value))
    except:
        return '{}'


@register.filter
def get_item(dictionary, key):
    """
    Get item from dictionary
    Usage: {{ dict|get_item:key }}
    """
    try:
        return dictionary.get(key)
    except:
        return None


@register.filter
def multiply(value, multiplier):
    """
    Multiply value by multiplier
    Usage: {{ value|multiply:100 }}
    """
    try:
        return float(value) * float(multiplier)
    except (ValueError, TypeError):
        return value


@register.filter
def percentage(value, total):
    """
    Calculate percentage
    Usage: {{ value|percentage:total }}
    """
    try:
        if float(total) == 0:
            return 0
        return (float(value) / float(total)) * 100
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.simple_tag
def bootstrap_color_cycle(index):
    """
    Get bootstrap color based on index
    Usage: {% bootstrap_color_cycle forloop.counter0 %}
    """
    colors = ['primary', 'success', 'warning', 'danger', 'info', 'secondary']
    return colors[int(index) % len(colors)]


@register.simple_tag
def word_color_class(word):
    """
    Get CSS class for word colors in P300
    Usage: {% word_color_class "green" %}
    """
    color_map = {
        'green': 'success',
        'red': 'danger', 
        'blue': 'primary',
        'yellow': 'warning',
        'purple': 'info',
        'silence': 'secondary',
        'XXXXX': 'secondary'
    }
    return color_map.get(str(word).lower(), 'secondary')


@register.filter
def dict_get(dictionary, key):
    """
    Get value from dictionary by key
    Usage: {{ my_dict|dict_get:"key" }}
    """
    try:
        return dictionary[key]
    except (KeyError, TypeError):
        return ''


@register.filter
def length_is(value, length):
    """
    Check if value length equals given length
    Usage: {{ my_list|length_is:5 }}
    """
    try:
        return len(value) == int(length)
    except (TypeError, ValueError):
        return False


@register.filter
def index(sequence, position):
    """
    Get item at index position
    Usage: {{ my_list|index:0 }}
    """
    try:
        return sequence[int(position)]
    except (IndexError, TypeError, ValueError):
        return None


@register.simple_tag
def confidence_color(confidence):
    """
    Get color class based on confidence level
    Usage: {% confidence_color 0.85 %}
    """
    try:
        conf = float(confidence)
        if conf >= 0.8:
            return 'success'
        elif conf >= 0.6:
            return 'warning' 
        elif conf >= 0.4:
            return 'info'
        else:
            return 'danger'
    except (ValueError, TypeError):
        return 'secondary'


@register.filter
def safe_divide(value, divisor):
    """
    Safe division that handles division by zero
    Usage: {{ value|safe_divide:divisor }}
    """
    try:
        divisor_val = float(divisor)
        if divisor_val == 0:
            return 0
        return float(value) / divisor_val
    except (ValueError, TypeError):
        return 0


@register.filter
def format_duration(seconds):
    """
    Format duration in seconds to readable format
    Usage: {{ 3661|format_duration }}  -> "1h 1m 1s"
    """
    try:
        total_seconds = int(float(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    except (ValueError, TypeError):
        return "0s"