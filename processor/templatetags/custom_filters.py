# processor/templatetags/custom_filters.py
import os
from django import template

register = template.Library()

@register.filter
def basename(value):
    """Get the basename of a path."""
    return os.path.basename(value)