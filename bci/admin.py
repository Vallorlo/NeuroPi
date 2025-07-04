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
    model = Prediction
    extra = 0
    readonly_fields = ['predicted_class', 'predicted_label', 'confidence', 'probabilities', 'prediction_time_ms', 'timestamp']
    fields = ['predicted_label', 'confidence', 'prediction_time_ms', 'timestamp']
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(PredictionSession)
class PredictionSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'model', 'status', 'started_at', 'stopped_at']
    list_filter = ['status', 'started_at', 'user']
    search_fields = ['name', 'user__username', 'model__name']
    readonly_fields = ['id', 'created_at']
    inlines = [PredictionInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'name', 'model')
        }),
        ('Session Parameters', {
            'fields': ('prediction_interval', 'window_duration')
        }),
        ('Status', {
            'fields': ('status', 'started_at', 'stopped_at', 'created_at')
        })
    )


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['session', 'predicted_label', 'confidence', 'prediction_time_ms', 'timestamp']
    list_filter = ['predicted_label', 'timestamp', 'session__user']
    search_fields = ['session__name', 'predicted_label']
    readonly_fields = ['id', 'timestamp']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'session', 'timestamp')
        }),
        ('Prediction Results', {
            'fields': ('predicted_class', 'predicted_label', 'confidence', 'probabilities')
        }),
        ('Performance', {
            'fields': ('prediction_time_ms',)
        })
    )


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    list_display = ['user', 'eeg_device', 'default_prediction_interval', 'updated_at']
    list_filter = ['eeg_device', 'updated_at']
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Hardware Settings', {
            'fields': ('eeg_device',)
        }),
        ('Default Prediction Settings', {
            'fields': ('default_prediction_interval', 'default_window_duration')
        }),
        ('UI Preferences', {
            'fields': ('show_confidence_threshold', 'max_prediction_history')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )