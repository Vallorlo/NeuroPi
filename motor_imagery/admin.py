# motor_imagery/admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import MotorImagerySession, MotorImageryTrial

class MotorImageryTrialInline(admin.TabularInline):
    model = MotorImageryTrial
    extra = 0
    readonly_fields = ['cue_start_time', 'imagery_start_time', 'trial_end_time']
    fields = ['trial_number', 'imagery_class', 'completed', 'cue_start_time', 'imagery_start_time', 'trial_end_time']

@admin.register(MotorImagerySession)
class MotorImagerySessionAdmin(admin.ModelAdmin):
    list_display = [
        'participant_name', 
        'session_name', 
        'is_completed', 
        'started_at', 
        'trial_count',
        'data_location',
        'unified_session_link'
    ]
    list_filter = ['is_completed', 'started_at', 'participant_name']
    search_fields = ['participant_name', 'session_name']
    readonly_fields = [
        'started_at', 
        'completed_at', 
        'total_trials', 
        'estimated_duration_minutes',
        'consistent_data_path',
        'data_folder_path'
    ]
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
        ('Data Storage', {
            'fields': ('eeg_data_file', 'data_folder_path', 'consistent_data_path'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('started_at', 'completed_at'),
            'classes': ('collapse',)
        })
    )
    
    def trial_count(self, obj):
        return obj.trials.count()
    trial_count.short_description = 'Completed Trials'
    
    def data_location(self, obj):
        """Show where the data is stored."""
        if obj.eeg_data_file:
            return format_html('<span style="color: green;">✓ Data Saved</span>')
        return format_html('<span style="color: orange;">⚠ No Data File</span>')
    data_location.short_description = 'Data Status'
    
    def unified_session_link(self, obj):
        """Link to unified session if it exists."""
        try:
            from trials.models import UnifiedTrialSession
            try:
                unified = UnifiedTrialSession.objects.get(
                    participant_name=obj.participant_name,
                    trial_type='motor_imagery',
                    started_at=obj.started_at
                )
                url = reverse('admin:trials_unifiedtrialsession_change', args=[unified.pk])
                return format_html('<a href="{}">View Unified</a>', url)
            except UnifiedTrialSession.DoesNotExist:
                return format_html('<span style="color: gray;">None</span>')
        except ImportError:
            return format_html('<span style="color: gray;">N/A</span>')
    unified_session_link.short_description = 'Unified Session'
    
    actions = ['create_unified_sessions', 'update_data_paths']
    
    def create_unified_sessions(self, request, queryset):
        """Admin action to create unified sessions for selected motor imagery sessions."""
        created_count = 0
        for session in queryset:
            try:
                unified = session.create_unified_session()
                if unified:
                    created_count += 1
            except Exception as e:
                self.message_user(request, f"Error creating unified session for {session}: {e}", level='ERROR')
        
        if created_count > 0:
            self.message_user(request, f"Successfully created {created_count} unified sessions.")
        else:
            self.message_user(request, "No unified sessions were created.", level='WARNING')
    
    create_unified_sessions.short_description = "Create unified sessions for selected sessions"
    
    def update_data_paths(self, request, queryset):
        """Admin action to update data folder paths to consistent structure."""
        updated_count = 0
        for session in queryset:
            old_path = session.data_folder_path
            session.data_folder_path = session.consistent_data_path
            session.save()
            updated_count += 1
        
        self.message_user(request, f"Updated data paths for {updated_count} sessions.")
    
    update_data_paths.short_description = "Update data paths to consistent structure"

@admin.register(MotorImageryTrial)
class MotorImageryTrialAdmin(admin.ModelAdmin):
    list_display = [
        'session', 
        'trial_number', 
        'imagery_class', 
        'completed', 
        'cue_start_time',
        'trial_duration'
    ]
    list_filter = ['imagery_class', 'completed', 'session__participant_name']
    search_fields = ['session__participant_name', 'session__session_name']
    readonly_fields = ['cue_start_time', 'imagery_start_time', 'trial_end_time', 'trial_duration']
    
    fieldsets = (
        ('Trial Information', {
            'fields': ('session', 'trial_number', 'imagery_class', 'completed')
        }),
        ('Timing', {
            'fields': ('cue_start_time', 'imagery_start_time', 'trial_end_time', 'trial_duration'),
            'classes': ('collapse',)
        })
    )
    
    def trial_duration(self, obj):
        """Calculate and display trial duration."""
        if obj.cue_start_time and obj.trial_end_time:
            duration = obj.trial_end_time - obj.cue_start_time
            return f"{duration.total_seconds():.1f}s"
        return "N/A"
    trial_duration.short_description = 'Total Duration'
    
    actions = ['create_unified_events']
    
    def create_unified_events(self, request, queryset):
        """Admin action to create unified events for selected trials."""
        created_count = 0
        for trial in queryset:
            try:
                trial.create_unified_events()
                created_count += 1
            except Exception as e:
                self.message_user(request, f"Error creating unified events for {trial}: {e}", level='ERROR')
        
        if created_count > 0:
            self.message_user(request, f"Successfully created unified events for {created_count} trials.")
        else:
            self.message_user(request, "No unified events were created.", level='WARNING')
    
    create_unified_events.short_description = "Create unified events for selected trials"