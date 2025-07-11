# speller/admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
import json

from .models import SpellerSession, SpellerEvent


@admin.register(SpellerSession)
class SpellerSessionAdmin(admin.ModelAdmin):
    """Admin interface for Speller Sessions"""
    
    list_display = [
        'name', 
        'user', 
        'status', 
        'motor_imagery_model_name', 
        'p300_model_name',
        'total_letters',
        'text_preview', 
        'created_at',
        'session_actions'
    ]
    
    list_filter = [
        'status', 
        'created_at', 
        'started_at', 
        'motor_imagery_model__approach',
        'p300_model__approach'
    ]
    
    search_fields = [
        'name', 
        'user__username', 
        'user__email',
        'motor_imagery_model__name',
        'p300_model__name'
    ]
    
    readonly_fields = [
        'id', 
        'created_at', 
        'started_at', 
        'stopped_at',
        'formatted_config',
        'session_statistics'
    ]
    
    fieldsets = [
        ('Basic Information', {
            'fields': ('id', 'name', 'user', 'status')
        }),
        ('Models', {
            'fields': ('motor_imagery_model', 'p300_model')
        }),
        ('Configuration', {
            'fields': ('formatted_config',),
            'classes': ('collapse',)
        }),
        ('Session Data', {
            'fields': ('current_text', 'selection_index')
        }),
        ('Statistics', {
            'fields': ('session_statistics',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'stopped_at'),
            'classes': ('collapse',)
        })
    ]
    
    def motor_imagery_model_name(self, obj):
        """Display motor imagery model name"""
        return obj.motor_imagery_model.name if obj.motor_imagery_model else '-'
    motor_imagery_model_name.short_description = 'MI Model'
    
    def p300_model_name(self, obj):
        """Display P300 model name"""
        return obj.p300_model.name if obj.p300_model else '-'
    p300_model_name.short_description = 'P300 Model'
    
    def text_preview(self, obj):
        """Display text preview"""
        if obj.current_text:
            preview = obj.current_text[:50]
            if len(obj.current_text) > 50:
                preview += '...'
            return format_html('<code>{}</code>', preview)
        return '-'
    text_preview.short_description = 'Text Preview'
    
    def session_actions(self, obj):
        """Display action buttons"""
        detail_url = reverse('speller:session_detail', args=[obj.pk])
        return format_html(
            '<a href="{}" class="button">View Details</a>',
            detail_url
        )
    session_actions.short_description = 'Actions'
    
    def formatted_config(self, obj):
        """Display formatted configuration"""
        config = {
            'Left Letters': obj.left_side_letters,
            'Right Letters': obj.right_side_letters,
            'Vocabulary': obj.vocabulary_words,
            'Total Letters': obj.total_letters
        }
        return format_html('<pre>{}</pre>', json.dumps(config, indent=2))
    formatted_config.short_description = 'Configuration'
    
    def session_statistics(self, obj):
        """Display session statistics"""
        events = obj.events.all()
        stats = {
            'Total Events': events.count(),
            'Motor Predictions': events.filter(event_type='MOTOR_PREDICTION').count(),
            'P300 Confirmations': events.filter(event_type='P300_CONFIRMATION').count(),
            'Letters Selected': events.filter(event_type='LETTER_SELECTED').count(),
            'Words Completed': events.filter(event_type='WORD_COMPLETED').count(),
            'Spaces Inserted': events.filter(event_type='SPACE_INSERTED').count(),
        }
        return format_html('<pre>{}</pre>', json.dumps(stats, indent=2))
    session_statistics.short_description = 'Statistics'


class SpellerEventInline(admin.TabularInline):
    """Inline admin for Speller Events"""
    model = SpellerEvent
    extra = 0
    readonly_fields = ['id', 'timestamp', 'event_type', 'confidence', 'formatted_event_data']
    fields = ['timestamp', 'event_type', 'confidence', 'formatted_event_data']
    
    def formatted_event_data(self, obj):
        """Display formatted event data"""
        if obj.event_data:
            return format_html('<pre>{}</pre>', json.dumps(obj.event_data, indent=2))
        return '-'
    formatted_event_data.short_description = 'Event Data'
    
    def has_add_permission(self, request, obj=None):
        return False  # Don't allow adding events manually


@admin.register(SpellerEvent)
class SpellerEventAdmin(admin.ModelAdmin):
    """Admin interface for Speller Events"""
    
    list_display = [
        'timestamp',
        'session_name',
        'event_type',
        'confidence_display',
        'event_summary'
    ]
    
    list_filter = [
        'event_type',
        'timestamp',
        'session__status',
        'session__user'
    ]
    
    search_fields = [
        'session__name',
        'session__user__username',
        'event_type'
    ]
    
    readonly_fields = [
        'id',
        'timestamp',
        'formatted_event_data'
    ]
    
    fieldsets = [
        ('Basic Information', {
            'fields': ('id', 'session', 'event_type', 'timestamp')
        }),
        ('Event Data', {
            'fields': ('confidence', 'formatted_event_data')
        })
    ]
    
    def session_name(self, obj):
        """Display session name"""
        return obj.session.name
    session_name.short_description = 'Session'
    session_name.admin_order_field = 'session__name'
    
    def confidence_display(self, obj):
        """Display confidence with formatting"""
        if obj.confidence is not None:
            return f"{obj.confidence:.1f}%"
        return '-'
    confidence_display.short_description = 'Confidence'
    confidence_display.admin_order_field = 'confidence'
    
    def event_summary(self, obj):
        """Display event summary"""
        if obj.event_data:
            if obj.event_type == 'LETTER_SELECTED':
                return f"Letter: {obj.event_data.get('letter', 'Unknown')}"
            elif obj.event_type == 'WORD_COMPLETED':
                return f"Word: {obj.event_data.get('completed_word', 'Unknown')}"
            elif obj.event_type == 'MOTOR_PREDICTION':
                return f"Prediction: {obj.event_data.get('predicted_label', 'Unknown')}"
            elif obj.event_type == 'SPACE_INSERTED':
                return "Space inserted"
            elif obj.event_type == 'WINDOW_MOVED':
                return f"Window moved: {obj.event_data.get('direction', 'Unknown')}"
        return '-'
    event_summary.short_description = 'Summary'
    
    def formatted_event_data(self, obj):
        """Display formatted event data"""
        if obj.event_data:
            return format_html('<pre>{}</pre>', json.dumps(obj.event_data, indent=2))
        return '-'
    formatted_event_data.short_description = 'Event Data'
    
    def has_add_permission(self, request):
        return False  # Don't allow adding events manually


# Add SpellerEvent inline to SpellerSession admin
SpellerSessionAdmin.inlines = [SpellerEventInline]