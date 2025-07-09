from django.contrib import admin
from .models import CommunicationSession, CommunicationEvent


@admin.register(CommunicationSession)
class CommunicationSessionAdmin(admin.ModelAdmin):
    list_display = ['session_name', 'user', 'communication_state', 'is_active', 'created_at']
    list_filter = ['communication_state', 'is_active', 'created_at']
    search_fields = ['session_name', 'user__username']
    readonly_fields = ['created_at', 'last_activity']


@admin.register(CommunicationEvent)
class CommunicationEventAdmin(admin.ModelAdmin):
    list_display = ['session', 'event_type', 'timestamp', 'confidence', 'selected_letter']
    list_filter = ['event_type', 'timestamp']
    search_fields = ['session__session_name']
    readonly_fields = ['timestamp']