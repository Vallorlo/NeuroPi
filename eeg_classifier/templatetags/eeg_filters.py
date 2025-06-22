from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary in template"""
    return dictionary.get(key)

@register.filter
def mul(value, arg):
    """Multiply filter"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def div(value, arg):
    """Divide filter"""
    try:
        return float(value) / float(arg)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0
