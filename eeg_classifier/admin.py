# eeg_classifier/admin.py
# Django admin configuration for EEG classification application

from django.contrib import admin
from django.utils.html import format_html
from .models import (
    Dataset, ClassificationModel, TrainingSession, 
    PredictionSession, WordClass, ModelPerformance
)

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'participant_name', 'total_samples', 'n_channels', 
        'sampling_rate', 'processed', 'uploaded_at'
    ]
    list_filter = ['processed', 'n_channels', 'sampling_rate', 'uploaded_at']
    search_fields = ['name', 'participant_name', 'description']
    readonly_fields = [
        'total_samples', 'total_duration', 'sampling_rate', 
        'n_channels', 'word_counts', 'processed'
    ]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'participant_name', 'file_path')
        }),
        ('Dataset Statistics', {
            'fields': (
                'total_samples', 'total_duration', 'sampling_rate', 
                'n_channels', 'word_counts', 'processed'
            ),
            'classes': ('collapse',)
        })
    )
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing existing object
            return self.readonly_fields + ('file_path',)
        return self.readonly_fields

@admin.register(ClassificationModel)
class ClassificationModelAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'model_type', 'accuracy_display', 'n_classes', 
        'epochs_trained', 'is_active', 'created_at'
    ]
    list_filter = ['model_type', 'is_active', 'n_classes', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = [
        'accuracy', 'val_accuracy', 'f1_score', 'epochs_trained', 
        'training_time', 'created_at'
    ]
    filter_horizontal = ['datasets_used']
    
    fieldsets = (
        ('Model Information', {
            'fields': ('name', 'description', 'model_type', 'is_active')
        }),
        ('Model Files', {
            'fields': ('model_file', 'scaler_file', 'label_encoder_file')
        }),
        ('Configuration', {
            'fields': ('window_size', 'n_classes', 'n_channels')
        }),
        ('Performance Metrics', {
            'fields': (
                'accuracy', 'val_accuracy', 'f1_score', 
                'epochs_trained', 'training_time'
            ),
            'classes': ('collapse',)
        }),
        ('Training Data', {
            'fields': ('datasets_used',),
            'classes': ('collapse',)
        })
    )
    
    def accuracy_display(self, obj):
        if obj.accuracy > 0:
            color = 'green' if obj.accuracy >= 70 else 'orange' if obj.accuracy >= 60 else 'red'
            return format_html(
                '<span style="color: {};">{:.2f}%</span>',
                color, obj.accuracy
            )
        return '-'
    accuracy_display.short_description = 'Accuracy'
    
    actions = ['set_active', 'set_inactive']
    
    def set_active(self, request, queryset):
        # First deactivate all models
        ClassificationModel.objects.update(is_active=False)
        # Then activate selected models
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} model(s) set as active.')
    set_active.short_description = 'Set selected models as active'
    
    def set_inactive(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} model(s) set as inactive.')
    set_inactive.short_description = 'Set selected models as inactive'

@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'model_type', 'status', 'progress_display', 
        'current_epoch', 'started_at', 'completed_at'
    ]
    list_filter = ['status', 'model_type', 'started_at']
    search_fields = ['name']
    readonly_fields = [
        'status', 'progress', 'current_epoch', 'training_log', 
        'error_message', 'started_at', 'completed_at'
    ]
    filter_horizontal = ['datasets']
    
    fieldsets = (
        ('Session Information', {
            'fields': ('name', 'datasets', 'model_type')
        }),
        ('Training Parameters', {
            'fields': (
                'window_size', 'overlap', 'epochs', 
                'batch_size', 'learning_rate'
            )
        }),
        ('Session Status', {
            'fields': (
                'status', 'progress', 'current_epoch', 
                'started_at', 'completed_at', 'final_model'
            ),
            'classes': ('collapse',)
        }),
        ('Logs and Errors', {
            'fields': ('training_log', 'error_message'),
            'classes': ('collapse',)
        })
    )
    
    def progress_display(self, obj):
        if obj.progress > 0:
            return format_html(
                '<div style="width: 100px; background-color: #f0f0f0; border-radius: 3px;">'
                '<div style="width: {}px; height: 20px; background-color: #007cba; border-radius: 3px;"></div>'
                '</div> {}%',
                obj.progress, obj.progress
            )
        return '0%'
    progress_display.short_description = 'Progress'

@admin.register(PredictionSession)
class PredictionSessionAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'model', 'dataset', 'accuracy_display', 
        'total_predictions', 'created_at'
    ]
    list_filter = ['model', 'created_at']
    search_fields = ['name']
    readonly_fields = ['predictions', 'accuracy_metrics', 'created_at']
    
    fieldsets = (
        ('Session Information', {
            'fields': ('name', 'model', 'dataset')
        }),
        ('Prediction Parameters', {
            'fields': ('window_duration', 'confidence_threshold')
        }),
        ('Results', {
            'fields': ('predictions', 'accuracy_metrics'),
            'classes': ('collapse',)
        })
    )
    
    def accuracy_display(self, obj):
        if obj.accuracy_metrics and 'overall_accuracy' in obj.accuracy_metrics:
            acc = obj.accuracy_metrics['overall_accuracy'] * 100
            color = 'green' if acc >= 70 else 'orange' if acc >= 60 else 'red'
            return format_html(
                '<span style="color: {};">{:.2f}%</span>',
                color, acc
            )
        return '-'
    accuracy_display.short_description = 'Accuracy'
    
    def total_predictions(self, obj):
        return len(obj.predictions) if obj.predictions else 0
    total_predictions.short_description = 'Total Predictions'

@admin.register(WordClass)
class WordClassAdmin(admin.ModelAdmin):
    list_display = [
        'word', 'brain_region', 'optimal_duration', 
        'difficulty_level', 'is_active'
    ]
    list_filter = ['brain_region', 'difficulty_level', 'is_active']
    search_fields = ['word', 'description', 'brain_region']
    
    fieldsets = (
        ('Word Information', {
            'fields': ('word', 'description', 'instructions')
        }),
        ('Neurological Details', {
            'fields': ('brain_region', 'key_channels')
        }),
        ('Classification Parameters', {
            'fields': ('optimal_duration', 'difficulty_level', 'is_active')
        })
    )

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model', 'created_at']
    list_filter = ['created_at']
    readonly_fields = [
        'confusion_matrix', 'per_class_precision', 'per_class_recall', 
        'per_class_f1', 'training_loss', 'training_accuracy', 
        'validation_loss', 'validation_accuracy', 'created_at'
    ]
    
    fieldsets = (
        ('Model', {
            'fields': ('model',)
        }),
        ('Classification Metrics', {
            'fields': (
                'confusion_matrix', 'per_class_precision', 
                'per_class_recall', 'per_class_f1'
            ),
            'classes': ('collapse',)
        }),
        ('Training Curves', {
            'fields': (
                'training_loss', 'training_accuracy', 
                'validation_loss', 'validation_accuracy'
            ),
            'classes': ('collapse',)
        })
    )

# Custom admin site configuration
admin.site.site_header = "EEG Classifier Admin"
admin.site.site_title = "EEG Classifier"
admin.site.index_title = "EEG Classification Management"