# trials/admin.py
# Django admin configuration for trials app models

from django.contrib import admin
from .models import Trial, WordSet, WordSetItem, VisualTrialSession, VisualTrialEvent

@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ['word', 'stage', 'date']
    list_filter = ['stage', 'date']
    search_fields = ['word']

class WordSetItemInline(admin.TabularInline):
    model = WordSetItem
    extra = 3
    fields = ['word', 'order']

@admin.register(WordSet)
class WordSetAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'word_count', 'created_date']
    list_filter = ['is_active', 'created_date']
    search_fields = ['name', 'description']
    inlines = [WordSetItemInline]
    
    def word_count(self, obj):
        return obj.words.count()
    word_count.short_description = 'Number of Words'

@admin.register(WordSetItem)
class WordSetItemAdmin(admin.ModelAdmin):
    list_display = ['word_set', 'word', 'order']
    list_filter = ['word_set']
    search_fields = ['word']

class VisualTrialEventInline(admin.TabularInline):
    model = VisualTrialEvent
    extra = 0
    readonly_fields = ['timestamp']
    fields = ['word', 'event_type', 'duration', 'timestamp']

@admin.register(VisualTrialSession)
class VisualTrialSessionAdmin(admin.ModelAdmin):
    list_display = ['participant_name', 'word_set', 'is_completed', 'started_at', 'total_events']
    list_filter = ['is_completed', 'word_set', 'started_at']
    search_fields = ['participant_name']
    readonly_fields = ['started_at', 'completed_at']
    inlines = [VisualTrialEventInline]
    
    fieldsets = (
        ('Session Information', {
            'fields': ('participant_name', 'word_set', 'is_completed')
        }),
        ('Configuration', {
            'fields': ('word_display_duration', 'rest_duration', 'repetitions_per_word')
        }),
        ('Timestamps', {
            'fields': ('started_at', 'completed_at'),
            'classes': ('collapse',)
        })
    )
    
    def total_events(self, obj):
        return obj.events.count()
    total_events.short_description = 'Total Events'

@admin.register(VisualTrialEvent)
class VisualTrialEventAdmin(admin.ModelAdmin):
    list_display = ['session', 'word', 'event_type', 'duration', 'timestamp']
    list_filter = ['event_type', 'timestamp', 'session__word_set']
    search_fields = ['word', 'session__participant_name']
    readonly_fields = ['timestamp']
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.select_related('session', 'session__word_set')