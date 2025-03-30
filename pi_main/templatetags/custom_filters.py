from django import template
import os
register = template.Library()

@register.filter
def split(value, arg):
    """Split a string into a list on the specified delimiter"""
    return value.split(arg)

@register.filter
def mul(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return None
    

@register.filter
def basename(value):
    """Get the basename of a path."""
    return os.path.basename(value)

@register.filter
def sub(value, arg):
    """Subtract the argument from the value"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return None
    

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary with the given key."""
    return dictionary.get(key)


# Alternative implementation that doesn't require the same length
@register.filter
def zip_lists(value, arg):
    """Zip two lists together, like Python's zip function."""
    return [(x, y) for x, y in zip(value or [], arg or [])]