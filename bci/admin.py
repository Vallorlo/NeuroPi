# bci/admin.py
"""
Fixed BCI Admin - Updated for current model structure
"""

from django.contrib import admin
from .models import (
    SessionData, TrainedModel, PredictionSession, 
    Prediction, SystemConfiguration
)


@admin.register(SessionData)
class SessionDataAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'approach', 'total_samples', 'created_at']
    list_filter = ['approach', 'created_at', 'user']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['id', 'channels', 'classes', 'total_samples', 'created_at', 'updated_at']
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'name', 'description', 'approach')
        }),
        ('File Information', {
            'fields': ('session_file', 'sampling_rate')
        }),
        ('Data Analysis', {
            'fields': ('channels', 'classes', 'total_samples'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'approach', 'status', 'validation_accuracy', 'is_active', 'created_at']
    list_filter = ['approach', 'status', 'is_active', 'created_at', 'user']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at']
    filter_horizontal = ['training_sessions']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'name', 'description', 'approach', 'status', 'is_active')
        }),
        ('Model Files', {
            'fields': ('model_file', 'scaler_file')
        }),
        ('Model Configuration', {
            'fields': ('n_classes', 'class_labels', 'channels', 'sampling_rate', 'window_duration', 'model_config'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('validation_accuracy', 'cross_val_mean', 'cross_val_std', 'training_epochs'),
            'classes': ('collapse',)
        }),
        ('Training Data', {
            'fields': ('training_sessions',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    actions = ['activate_models', 'deactivate_models']

    def activate_models(self, request, queryset):
        for model in queryset.filter(status='completed'):
            model.activate()
        self.message_user(request, f"Activated {queryset.count()} models.")
    activate_models.short_description = "Activate selected models"

    def deactivate_models(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f"Deactivated {queryset.count()} models.")
    deactivate_models.short_description = "Deactivate selected models"


class PredictionInline(admin.TabularInline):
    """Inline for predictions in prediction session admin"""
    model = Prediction
    extra = 0
    readonly_fields = ['id', 'predicted_class', 'confidence', 'timestamp']
    fields = ['id', 'predicted_class', 'confidence', 'timestamp']
    can_delete = False


@admin.register(PredictionSession)
class PredictionSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'model', 'status', 'created_at']  # Removed 'approach'
    list_filter = ['status', 'created_at', 'user', 'model__approach']  # Use model__approach
    search_fields = ['name', 'user__username', 'model__name']  # Removed 'description'
    readonly_fields = ['id', 'created_at', 'started_at', 'stopped_at']
    inlines = [PredictionInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'name', 'model')  # Removed non-existent fields
        }),
        ('Session Parameters', {
            'fields': ('prediction_interval', 'window_duration')  # Original fields
        }),
        ('Status', {
            'fields': ('status', 'started_at', 'stopped_at')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'model')


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['session', 'predicted_class', 'confidence', 'timestamp']
    list_filter = ['predicted_class', 'session__model__approach', 'timestamp']  # FIXED
    search_fields = ['session__name', 'predicted_class']
    readonly_fields = ['id', 'timestamp']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'session', 'predicted_class', 'confidence')
        }),
        ('Prediction Data', {
            'fields': ('probabilities', 'prediction_time_ms'),  # Only existing fields
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('timestamp',)
        })
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session', 'session__model')


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    list_display = ['user', 'eeg_device', 'default_prediction_interval', 'updated_at']  # FIXED
    list_filter = ['eeg_device', 'updated_at']  # Removed non-existent fields
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('EEG Device Configuration', {
            'fields': ('eeg_device',)  # Only existing fields
        }),
        ('Default Settings', {
            'fields': ('default_prediction_interval', 'default_window_duration')
        }),
        ('UI Configuration', {
            'fields': ('show_confidence_threshold', 'max_prediction_history')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def has_add_permission(self, request):
        # Only allow one configuration per user
        if request.user.is_superuser:
            return True
        return not SystemConfiguration.objects.filter(user=request.user).exists()


# Additional admin customizations
admin.site.site_header = "NeuroPi BCI Administration"
admin.site.site_title = "NeuroPi BCI Admin"
admin.site.index_title = "Welcome to NeuroPi BCI Administration"