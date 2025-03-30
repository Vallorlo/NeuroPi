import os
from django import template

register = template.Library()

@register.filter
def basename(value):
    """Get the basename of a path."""
    return os.path.basename(value)

@register.filter
def mul(value, arg):
    """Multiply the value by the argument."""
    return value * arg

@register.filter
def sub(value, arg):
    """Subtract the argument from the value."""
    return value - arg

@register.filter
def zip(value, arg):
    """Zip two lists together, like Python's zip function."""
    return zip_longest(value, arg) if value and arg else []

from itertools import zip_longest

# Alternative implementation that doesn't require the same length
@register.filter
def zip_lists(value, arg):
    """Zip two lists together, like Python's zip function."""
    return [(x, y) for x, y in zip(value or [], arg or [])]