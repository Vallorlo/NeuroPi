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
            'fields': ('accuracy', 'val_accuracy', 'f1_score', 'epochs_trained', 'training_time'),
            'classes': ('collapse',)
        }),
        ('Training Data', {
            'fields': ('datasets_used',),
            'classes': ('collapse',)
        })
    )
    
    def accuracy_display(self, obj):
        """Display accuracy with color coding"""
        if obj.accuracy >= 80:
            color = 'green'
        elif obj.accuracy >= 60:
            color = 'orange'
        else:
            color = 'red'
        return format_html(
            '<span style="color: {};">{:.1f}%</span>',
            color,
            obj.accuracy
        )
    accuracy_display.short_description = 'Accuracy'
    accuracy_display.admin_order_field = 'accuracy'

@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'status', 'progress', 'model_type', 'current_epoch', 
        'epochs', 'started_at'
    ]
    list_filter = ['status', 'model_type', 'started_at']
    search_fields = ['name']
    readonly_fields = [
        'status', 'progress', 'current_epoch', 'final_model', 
        'training_log', 'error_message', 'started_at', 'completed_at'
    ]
    filter_horizontal = ['datasets']
    
    fieldsets = (
        ('Session Information', {
            'fields': ('name', 'datasets', 'model_type')
        }),
        ('Training Parameters', {
            'fields': ('window_size', 'overlap', 'epochs', 'batch_size', 'learning_rate')
        }),
        ('Session Status', {
            'fields': ('status', 'progress', 'current_epoch', 'final_model'),
            'classes': ('collapse',)
        }),
        ('Logs and Errors', {
            'fields': ('training_log', 'error_message', 'started_at', 'completed_at'),
            'classes': ('collapse',)
        })
    )

@admin.register(PredictionSession)
class PredictionSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'model', 'dataset', 'window_duration', 'created_at']
    list_filter = ['created_at', 'model__model_type']
    search_fields = ['name', 'model__name', 'dataset__name']
    readonly_fields = ['predictions', 'accuracy_metrics', 'created_at']

@admin.register(WordClass)
class WordClassAdmin(admin.ModelAdmin):
    list_display = [
        'word', 'brain_region', 'optimal_duration', 'difficulty_level', 
        'is_active', 'created_at'
    ]
    list_filter = ['is_active', 'difficulty_level', 'created_at']
    search_fields = ['word', 'description', 'brain_region']
    
    fieldsets = (
        ('Word Information', {
            'fields': ('word', 'description', 'is_active')
        }),
        ('Neuroscience Details', {
            'fields': ('brain_region', 'key_channels', 'instructions')
        }),
        ('Classification Parameters', {
            'fields': ('optimal_duration', 'difficulty_level')
        })
    )

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model', 'created_at']
    list_filter = ['created_at', 'model__model_type']
    search_fields = ['model__name']
    readonly_fields = [
        'confusion_matrix', 'per_class_precision', 'per_class_recall', 
        'per_class_f1', 'training_loss', 'training_accuracy',
        'validation_loss', 'validation_accuracy', 'created_at'
    ]
