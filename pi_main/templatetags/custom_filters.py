from django import template

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
def sub(value, arg):
    """Subtract the argument from the value"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return None