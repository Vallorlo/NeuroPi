from django.contrib import admin
from .models import MotorImagerySession, MotorImageryTrial

class MotorImageryTrialInline(admin.TabularInline):
    model = MotorImageryTrial
    extra = 0
    readonly_fields = ['cue_start_time', 'imagery_start_time', 'trial_end_time']
    fields = ['trial_number', 'imagery_class', 'completed', 'cue_start_time', 'imagery_start_time', 'trial_end_time']

@admin.register(MotorImagerySession)
class MotorImagerySessionAdmin(admin.ModelAdmin):
    list_display = ['participant_name', 'session_name', 'is_completed', 'started_at', 'trial_count']
    list_filter = ['is_completed', 'started_at', 'participant_name']
    search_fields = ['participant_name', 'session_name']
    readonly_fields = ['started_at', 'completed_at', 'total_trials', 'estimated_duration_minutes']
    inlines = [MotorImageryTrialInline]
    
    fieldsets = (
        ('Session Information', {
            'fields': ('participant_name', 'session_name', 'is_completed')
        }),
        ('Configuration', {
            'fields': ('imagery_duration', 'cue_duration', 'rest_duration', 'trials_per_class')
        }),
        ('Calculated Values', {
            'fields': ('total_trials', 'estimated_duration_minutes'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('started_at', 'completed_at'),
            'classes': ('collapse',)
        }),
        ('Data', {
            'fields': ('eeg_data_file',),
            'classes': ('collapse',)
        })
    )
    
    def trial_count(self, obj):
        return obj.trials.count()
    trial_count.short_description = 'Completed Trials'

@admin.register(MotorImageryTrial)
class MotorImageryTrialAdmin(admin.ModelAdmin):
    list_display = ['session', 'trial_number', 'imagery_class', 'completed', 'cue_start_time']
    list_filter = ['imagery_class', 'completed', 'session__participant_name']
    search_fields = ['session__participant_name', 'session__session_name']
    readonly_fields = ['cue_start_time', 'imagery_start_time', 'trial_end_time']
